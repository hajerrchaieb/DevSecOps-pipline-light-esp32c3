"""
agents/autofix_agent.py  —  Agent 8 : Auto-Fix Agent
======================================================
ROLE:
  Reads issues from ALL previous agent reports (debug, code-review,
  security, fault-analysis), generates unified diff patches for each
  affected source file, and saves them into reports/patches/.

  On second CI run the patches are applied BEFORE the build step so
  the firmware is recompiled with the fixes in place.

INPUTS  (all read from reports/):
  debug-report-{target}.json
  code-review-{target}.json
  security-report-{target}.json
  fault-analysis-report-{target}.json

OUTPUTS:
  reports/autofix-report-{target}.json   ← consumed by orchestrator
  reports/patches/fix_<name>.patch       ← unified diff, git-apply ready
  reports/patches/APPLY_ALL.sh           ← convenience apply script

INTEGRATION WITH CI (ci.yml):
  The ci.yml workflow has two apply hooks:
    1. "Apply AutoFix patches" step BEFORE the second build (Stage 4b)
       runs:  bash reports/patches/APPLY_ALL.sh
    2. Artifact upload includes reports/patches/ so patches survive
       across workflow jobs.

USAGE:
  python3 agents/autofix_agent.py
  python3 agents/autofix_agent.py --apply     # also git-apply patches
  python3 agents/autofix_agent.py --target esp32c3
"""

import difflib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TARGET  = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS = Path("reports")
PATCHES = Path("reports/patches")

logging.basicConfig(level=logging.INFO, format="[AutoFix] %(message)s")
logger = logging.getLogger("autofix_agent")


# ═══════════════════════════════════════════════════════════════
# SOURCE RESOLUTION  (same multi-path strategy as other agents)
# ═══════════════════════════════════════════════════════════════

def _find_source_dir() -> Path | None:
    """Return the first existing ESP-Matter light/main source directory."""
    candidates = [
        Path("esp-matter/examples/light/main"),
        Path("../esp-matter/examples/light/main"),
        Path("/opt/espressif/esp-matter/examples/light/main"),
        Path.home() / "esp-matter/examples/light/main",
    ]
    for c in candidates:
        r = c.resolve()
        if r.is_dir() and (list(r.glob("*.cpp")) + list(r.glob("*.c"))):
            logger.info("Source dir: %s", r)
            return r
    logger.warning("Source dir not found — patches will be template-only")
    return None


def _read_source(filename: str, src_dir: Path | None) -> str:
    """Read a source file; return empty string if unavailable."""
    if src_dir is None:
        return ""
    p = src_dir / filename
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        return content
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
# ISSUE COLLECTION  (aggregates all agent reports)
# ═══════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_issues(target: str) -> List[Dict]:
    """
    Collect CRITICAL + HIGH issues from all previous agent reports.
    Returns a unified list of issue dicts, each with keys:
      file, description, severity, fix_hint, cwe, source
    """
    issues: List[Dict] = []

    # ── Agent 1: Debug — compilation errors ──────────────────────
    debug = _load_json(REPORTS / f"debug-report-{target}.json")
    for err in debug.get("compilation_errors", []):
        issues.append({
            "source":      "debug_agent",
            "severity":    "critical",
            "file":        err.get("file", "app_main.cpp"),
            "location":    str(err.get("line", "?")),
            "description": err.get("error", err.get("root_cause", "?")),
            "fix_hint":    err.get("fix", ""),
            "cwe":         "",
        })
    # Warnings that are high-impact
    for w in debug.get("warnings", []):
        if w.get("impact") == "high":
            issues.append({
                "source":      "debug_agent",
                "severity":    "high",
                "file":        "app_main.cpp",
                "location":    "see warning",
                "description": w.get("description", ""),
                "fix_hint":    "",
                "cwe":         "",
            })

    # ── Agent 3: Code Review — quality issues ─────────────────────
    review = _load_json(REPORTS / f"code-review-{target}.json")
    for issue in review.get("issues", []):
        if issue.get("severity", "low") in ("critical", "high"):
            issues.append({
                "source":      "code_review",
                "severity":    issue.get("severity", "high"),
                "file":        issue.get("file", "app_main.cpp"),
                "location":    issue.get("location", "?"),
                "description": issue.get("description", ""),
                "fix_hint":    issue.get("good_code", issue.get("fix", "")),
                "cwe":         issue.get("cwe", ""),
            })

    # ── Agent 2: Security — secrets + CVEs ───────────────────────
    security = _load_json(REPORTS / f"security-report-{target}.json")
    for secret in security.get("secrets_found", []):
        issues.append({
            "source":      "security_agent",
            "severity":    "critical",
            "file":        secret.get("file", "app_main.cpp"),
            "location":    "hardcoded credential",
            "description": f"Hardcoded {secret.get('type', 'secret')} detected",
            "fix_hint":    secret.get("action", "Move to NVS or env variable"),
            "cwe":         "CWE-798",
        })

    # ── Agent 7: Fault Analysis — robustness fixes ────────────────
    fault = _load_json(REPORTS / f"fault-analysis-report-{target}.json")
    for sc in fault.get("failed_scenarios_analysis", []):
        issues.append({
            "source":      "fault_analysis",
            "severity":    "high",
            "file":        sc.get("affected_file", "app_main.cpp"),
            "location":    sc.get("scenario", "see fault scenario"),
            "description": sc.get("root_cause", ""),
            "fix_hint":    sc.get("fix_code", ""),
            "cwe":         sc.get("cwe", ""),
        })

    logger.info("Collected %d issue(s) for %s", len(issues), target)
    return issues


# ═══════════════════════════════════════════════════════════════
# PATCH GENERATION  (LLM → unified diff)
# ═══════════════════════════════════════════════════════════════

def _generate_patch_for_file(
    filename: str,
    original_src: str,
    issues: List[Dict],
    llm,
) -> Dict | None:
    """
    Ask the LLM to produce a fixed version of `filename`, then compute
    a unified diff against the original.  Returns a patch dict or None
    if nothing changed.
    """
    issues_text = "\n".join(
        f"  Issue {i+1}: [{iss['severity'].upper()}] {iss['description']}\n"
        f"    Location : {iss.get('location', '?')}\n"
        f"    Fix hint : {iss.get('fix_hint', '')[:300]}\n"
        f"    CWE      : {iss.get('cwe', 'N/A')}"
        for i, iss in enumerate(issues)
    )

    # If we have no source, generate a comment-only patch as a reminder
    if not original_src:
        original_src = (
            f"// {filename} — source not available during patch generation\n"
            f"// Patch is a template; apply manually after obtaining source.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior ESP32/ESP-IDF firmware engineer.
Your task: fix the C++ source file below based on the listed issues.
Return ONLY the complete corrected file content — no explanations,
no markdown fences, no comments about what you changed.
The output will be diffed against the original to produce a patch.
Follow ESP-IDF conventions: esp_err_t returns, ESP_LOGE for errors,
NULL checks after every heap allocation or handle lookup."""),

        ("human", """=== FILE: {filename} ===

=== ISSUES TO FIX ===
{issues_text}

=== ORIGINAL SOURCE ===
{source}

Return the complete corrected file content only."""),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        fixed = chain.invoke({
            "filename":    filename,
            "issues_text": issues_text,
            "source":      original_src[:6000],
        })
    except Exception as e:
        logger.error("LLM call failed for %s: %s", filename, e)
        return None

    # Strip accidental markdown fences the model might add
    fixed = fixed.strip()
    for fence in ("```cpp", "```c", "```", "`"):
        if fixed.startswith(fence):
            fixed = fixed[len(fence):]
        if fixed.endswith(fence):
            fixed = fixed[:-len(fence)]
    fixed = fixed.strip()

    # Compute unified diff
    diff_lines = list(difflib.unified_diff(
        original_src.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=f"a/main/{filename}",
        tofile=f"b/main/{filename}",
        lineterm="\n",
    ))

    if not diff_lines:
        logger.info("No changes produced for %s — skipping patch", filename)
        return None

    patch_content = "".join(diff_lines)

    # Save .patch file
    PATCHES.mkdir(parents=True, exist_ok=True)
    safe_name  = filename.replace("/", "_").replace("\\", "_")
    patch_name = f"fix_{safe_name}"
    patch_path = PATCHES / f"{patch_name}.patch"
    patch_path.write_text(patch_content, encoding="utf-8")
    logger.info("Patch saved: %s (%d lines)", patch_path.name, len(diff_lines))

    return {
        "filename":      filename,
        "patch_file":    str(patch_path),
        "diff_lines":    len(diff_lines),
        "issues_fixed":  [iss["description"][:120] for iss in issues],
        "cwe_list":      list({iss["cwe"] for iss in issues if iss.get("cwe")}),
        "severity":      max((iss["severity"] for iss in issues),
                             key=lambda s: {"critical": 2, "high": 1}.get(s, 0)),
    }


# ═══════════════════════════════════════════════════════════════
# APPLY SCRIPT GENERATION
# ═══════════════════════════════════════════════════════════════

def _write_apply_script(patch_files: List[Path]) -> Path:
    """
    Generate reports/patches/APPLY_ALL.sh.
    The ci.yml Stage 4b step calls:   bash reports/patches/APPLY_ALL.sh
    """
    lines = [
        "#!/bin/bash",
        "# AutoFix — apply all generated patches to ESP-Matter source",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# Called by ci.yml Stage 4b BEFORE the second build:",
        "#   - name: 🩹 Apply AutoFix patches",
        "#     run: bash reports/patches/APPLY_ALL.sh",
        "#",
        "set -e",
        "",
        "# Resolve ESP-Matter source dir (same logic as agents)",
        "ESP_MAIN=''",
        "for CANDIDATE in \\",
        "    'esp-matter/examples/light/main' \\",
        "    '../esp-matter/examples/light/main' \\",
        "    '/opt/espressif/esp-matter/examples/light/main'; do",
        "  if [ -d \"$CANDIDATE\" ]; then ESP_MAIN=\"$CANDIDATE\"; break; fi",
        "done",
        "",
        "if [ -z \"$ESP_MAIN\" ]; then",
        "  echo '[AutoFix] ERROR: ESP-Matter source directory not found'",
        "  exit 1",
        "fi",
        "echo \"[AutoFix] Applying patches to: $ESP_MAIN\"",
        "",
        "PATCHES_DIR=\"$(dirname \"$0\")\"",
        "APPLIED=0",
        "SKIPPED=0",
        "",
    ]

    for pf in sorted(patch_files):
        lines += [
            f"echo '[AutoFix] → {pf.name}'",
            # First check if the patch applies cleanly
            f"if git apply --check --directory=\"$ESP_MAIN\" \"{pf.resolve()}\" 2>/dev/null; then",
            f"  git apply --directory=\"$ESP_MAIN\" \"{pf.resolve()}\"",
            "  APPLIED=$((APPLIED+1))",
            "  echo '  ✅ Applied'",
            "else",
            f"  echo '  ⚠️  Skipped (already applied or conflicts)'",
            "  SKIPPED=$((SKIPPED+1))",
            "fi",
            "",
        ]

    lines += [
        "echo \"\"",
        "echo \"[AutoFix] Done — Applied: $APPLIED | Skipped: $SKIPPED\"",
        "",
        "# Verify build still compiles after patches",
        "if command -v idf.py &>/dev/null; then",
        "  echo '[AutoFix] Verifying build...'",
        "  cd \"$ESP_MAIN\"",
        "  idf.py build 2>&1 | tail -8",
        "fi",
    ]

    script_path = PATCHES / "APPLY_ALL.sh"
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    logger.info("Apply script: %s", script_path)
    return script_path


# ═══════════════════════════════════════════════════════════════
# GENERATED TEST INTEGRATION
# ═══════════════════════════════════════════════════════════════

def _apply_generated_tests(target: str) -> Dict:
    """
    Copy the generated test file into the ESP-Matter test directory
    so the second CI run picks it up automatically.

    Returns a dict describing what was done (included in the report).
    """
    test_src = REPORTS / f"generated_tests_{target}.cpp"
    if not test_src.exists():
        return {"status": "no_test_file", "path": str(test_src)}

    src_dir = _find_source_dir()
    if src_dir is None:
        return {"status": "source_not_found"}

    # Target location: examples/light/test/
    test_dir = src_dir.parent / "test"
    test_dir.mkdir(exist_ok=True)
    dest = test_dir / f"test_light_app_{target}.cpp"

    try:
        content = test_src.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        logger.info("Test file deployed: %s", dest)
        return {
            "status":    "deployed",
            "source":    str(test_src),
            "dest":      str(dest),
            "size_chars": len(content),
        }
    except Exception as e:
        logger.error("Failed to deploy test file: %s", e)
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_autofix_agent(
    target:      str  = TARGET,
    apply_patches: bool = False,
) -> dict:
    """
    Run the Auto-Fix Agent.

    Args:
        target:        ESP32 chip target (e.g. "esp32c3")
        apply_patches: If True, immediately run APPLY_ALL.sh after generation

    Returns:
        Report dict written to reports/autofix-report-{target}.json
        Keys consumed by orchestrator:
          patches_generated (int)
          status            (str)
          test_integration  (dict)
    """
    print(f"\n{'='*60}")
    print(f"Agent 8 — Auto-Fix Agent")
    print(f"Target : {target}")
    print(f"{'='*60}")

    PATCHES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # ── 1. Collect issues ─────────────────────────────────────────
    issues = collect_issues(target)

    if not issues:
        logger.info("No critical/high issues found — nothing to patch")
        report = {
            "agent":             "autofix_agent",
            "target":            target,
            "timestamp":         datetime.now().isoformat(),
            "issues_analyzed":   0,
            "patches_generated": 0,
            "patch_files":       [],
            "apply_script":      str(PATCHES / "APPLY_ALL.sh"),
            "test_integration":  _apply_generated_tests(target),
            "status":            "clean",
            "summary":           "No critical or high issues found — no patches needed.",
        }
        _out = REPORTS / f"autofix-report-{target}.json"
        _out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    print(f"[AutoFix] {len(issues)} issue(s) to address")

    # ── 2. Group issues by file ───────────────────────────────────
    issues_by_file: Dict[str, List[Dict]] = {}
    for issue in issues:
        f = issue.get("file") or "app_main.cpp"
        # Normalise: strip path prefix, keep basename
        f = Path(f).name if "/" in f or "\\" in f else f
        issues_by_file.setdefault(f, []).append(issue)

    # ── 3. Load sources ───────────────────────────────────────────
    src_dir = _find_source_dir()
    sources: Dict[str, str] = {}
    for filename in issues_by_file:
        sources[filename] = _read_source(filename, src_dir)

    # ── 4. Instantiate LLM ───────────────────────────────────────
    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.05,
        max_tokens=4000,
    )

    # ── 5. Generate patches (max 6 files to stay in token budget) ─
    all_patch_results: List[Dict] = []
    for filename, file_issues in list(issues_by_file.items())[:6]:
        print(f"[AutoFix] Patching {filename} ({len(file_issues)} issue(s))...")
        patch = _generate_patch_for_file(
            filename=filename,
            original_src=sources.get(filename, ""),
            issues=file_issues,
            llm=llm,
        )
        if patch:
            all_patch_results.append(patch)
        else:
            logger.info("No patch produced for %s", filename)

    # ── 6. Write APPLY_ALL.sh ─────────────────────────────────────
    patch_files = list(PATCHES.glob("*.patch"))
    apply_script = _write_apply_script(patch_files)

    # ── 7. Deploy generated tests into ESP-Matter test dir ────────
    test_integration = _apply_generated_tests(target)
    if test_integration.get("status") == "deployed":
        print(f"[AutoFix] Test file deployed → {test_integration['dest']}")

    # ── 8. Optionally apply patches immediately ───────────────────
    apply_results: List[Dict] = []
    if apply_patches and patch_files:
        print("[AutoFix] Applying patches via git apply...")
        for pf in sorted(patch_files):
            try:
                check = subprocess.run(
                    ["git", "apply", "--check", str(pf)],
                    capture_output=True, text=True,
                )
                if check.returncode == 0:
                    subprocess.run(["git", "apply", str(pf)], check=True)
                    apply_results.append({"patch": pf.name, "status": "applied"})
                    print(f"  ✅ {pf.name}")
                else:
                    apply_results.append({
                        "patch":  pf.name,
                        "status": "skipped",
                        "reason": check.stderr[:200],
                    })
                    print(f"  ⚠️  {pf.name} — {check.stderr[:80]}")
            except Exception as e:
                apply_results.append({"patch": pf.name, "status": "error", "error": str(e)})
                print(f"  ❌ {pf.name} — {e}")

    # ── 9. Build final report ─────────────────────────────────────
    total_patches = len(all_patch_results)
    report = {
        "agent":             "autofix_agent",
        "target":            target,
        "timestamp":         datetime.now().isoformat(),
        "issues_analyzed":   len(issues),
        "files_patched":     len(issues_by_file),
        "patches_generated": total_patches,        # ← orchestrator reads this
        "patch_files":       [p["patch_file"] for p in all_patch_results],
        "apply_script":      str(apply_script),
        "patches_detail":    all_patch_results,
        "apply_results":     apply_results,
        "test_integration":  test_integration,
        "status":            (
            "patches_generated" if total_patches > 0 else "no_patches_generated"
        ),
        "summary": (
            f"{total_patches} patch(es) generated for "
            f"{len(issues_by_file)} file(s), addressing {len(issues)} issue(s). "
            f"Test integration: {test_integration.get('status', 'unknown')}. "
            f"Apply with: bash {apply_script}"
        ),
    }

    out = REPORTS / f"autofix-report-{target}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[AutoFix] {total_patches} patch(es) generated")
    print(f"[AutoFix] Apply script  : {apply_script}")
    print(f"[AutoFix] Report        : {out}")
    print(f"[AutoFix] Tests deployed: {test_integration.get('status')}")
    return report


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoFix Agent — generate + apply patches")
    parser.add_argument("--target",  default=TARGET,  help="ESP32 chip target")
    parser.add_argument("--apply",   action="store_true",
                        help="Apply patches immediately via git apply")
    args = parser.parse_args()
    result = run_autofix_agent(target=args.target, apply_patches=args.apply)
    print(f"\nAutoFix: {result['patches_generated']} patch(es) | status: {result['status']}")