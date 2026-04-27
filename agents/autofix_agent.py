"""
agents/autofix_agent.py — Agent 8 : AutoFix Agent
==================================================
ROLE:
  Reads the analysis reports produced by the OTHER agents
  (security, code-review, debug, fault-analysis) and generates
  unified-diff *.patch files that can be applied automatically
  by Stage 4b / Stage 4c in the SECOND CI run.

KEY BEHAVIOURS:
  1. Distinguishes between:
       (a) issues that live INSIDE the source code
           (app_main.cpp / app_driver.cpp / app_priv.h)
              -> generate a real .patch file for the second run.
       (b) issues that live OUTSIDE the source
           (CI config, dependencies, missing files, env vars,
            secret scanning, etc.)
              -> emit a textual "instruction" written into the
              autofix-report so a human can fix it manually.
  2. Saves every generated patch into  reports/patches/*.patch.
  3. Saves the consolidated report into
        reports/autofix-report-{target}.json
  4. Exposes the function `run_autofix_agent(target=...)`
     so the orchestrator can call it like every other agent.
     (The old standalone `run()` / `__main__` entry-point kept too,
      so the file can still be executed directly inside CI.)

INPUTS:
  reports/security-report-{target}.json
  reports/code-review-{target}.json
  reports/debug-report-{target}.json
  reports/fault-analysis-report-{target}.json
  esp-matter/examples/light/main/   (source files for context)

OUTPUTS:
  reports/patches/*.patch
  reports/autofix-report-{target}.json
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

# ── optional LLM (used only if GROQ_API_KEY available) ──────────────
try:
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    load_dotenv()
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
TARGET   = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS  = Path("reports")
PATCHES  = REPORTS / "patches"

# Where the actual ESP-Matter source files live in the repo
# (committed once via the `sync-esp-matter-source` workflow).
SOURCE_DIR_CANDIDATES = [
    Path("esp-matter/examples/light/main"),
    Path(os.getenv("EXAMPLE_PATH", "esp-matter/examples/light")) / "main",
    Path("/opt/espressif/esp-matter/examples/light/main"),
]

SOURCE_FILES = ["app_main.cpp", "app_driver.cpp", "app_priv.h"]

# Issue categories that CANNOT be fixed by a code patch
# -> produce instructions instead.
NON_CODE_KEYWORDS = (
    "ci",  "workflow", "pipeline", ".yml", ".yaml",
    "secret", "token", "api key", "groq_api_key", "pat_token",
    "dependency", "sbom", "package", "version pin",
    "docker", "image", "registry",
    "permission", "chmod", "ownership",
    "env", "environment variable",
)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_source_dir() -> Path | None:
    for c in SOURCE_DIR_CANDIDATES:
        if c.is_dir() and any((c / f).exists() for f in SOURCE_FILES):
            return c
    return None


def _read_source(src_dir: Path) -> dict:
    out = {}
    for fn in SOURCE_FILES:
        p = src_dir / fn
        if p.exists():
            out[fn] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def _is_code_issue(issue: dict) -> bool:
    """
    Decide whether the issue can be patched directly in the source code.
    """
    text = " ".join([
        str(issue.get("description", "")),
        str(issue.get("file", "")),
        str(issue.get("location", "")),
        str(issue.get("category", "")),
        str(issue.get("type", "")),
    ]).lower()

    # Explicit non-code markers win.
    if any(k in text for k in NON_CODE_KEYWORDS):
        return False

    # Anything that mentions one of our source files is a code issue.
    if any(fn in text for fn in SOURCE_FILES):
        return True

    # Default: if the issue has a code snippet field, treat as code.
    return bool(issue.get("code_snippet") or issue.get("suggested_fix"))


def _collect_issues(reports: dict) -> list[dict]:
    """
    Flatten findings from all agent reports into a single list of
    dicts with a uniform shape:
        { source_agent, severity, file, description, suggested_fix }
    """
    issues: list[dict] = []

    # Security agent
    sec = reports.get("security", {})
    for cve in sec.get("critical_cves", []) or []:
        issues.append({
            "source_agent": "security",
            "severity":     cve.get("severity", "high"),
            "file":         cve.get("file", ""),
            "description":  cve.get("description", str(cve)),
            "suggested_fix": cve.get("remediation", ""),
            "category":     "security",
        })
    for s in sec.get("secrets_found", []) or []:
        issues.append({
            "source_agent": "security",
            "severity":     "critical",
            "file":         s.get("file", ""),
            "description":  f"Secret detected: {s.get('rule', s)}",
            "suggested_fix": "Rotate the secret and remove it from VCS.",
            "category":     "secret",
        })

    # Code-review agent
    cr = reports.get("code_review", {})
    for it in (cr.get("issues") or cr.get("findings") or []):
        issues.append({
            "source_agent": "code_review",
            "severity":     it.get("severity", "medium"),
            "file":         it.get("file", ""),
            "description":  it.get("description") or it.get("issue", str(it)),
            "suggested_fix": it.get("suggested_fix") or it.get("fix", ""),
            "code_snippet": it.get("code_snippet", ""),
            "category":     "quality",
        })

    # Debug agent
    dbg = reports.get("debug", {})
    for it in (dbg.get("issues") or dbg.get("bugs") or []):
        issues.append({
            "source_agent": "debug",
            "severity":     it.get("severity", "medium"),
            "file":         it.get("file", ""),
            "description":  it.get("description") or str(it),
            "suggested_fix": it.get("suggested_fix") or it.get("fix", ""),
            "category":     "bug",
        })

    # Fault-analysis agent
    fa = reports.get("fault", {})
    for it in (fa.get("regressions") or fa.get("issues") or []):
        issues.append({
            "source_agent": "fault_analysis",
            "severity":     it.get("severity", "medium"),
            "file":         it.get("file", ""),
            "description":  it.get("description") or str(it),
            "suggested_fix": it.get("suggested_fix") or "",
            "category":     "robustness",
        })

    return issues


# ════════════════════════════════════════════════════════════════════
# PATCH GENERATION
# ════════════════════════════════════════════════════════════════════
def _make_unified_diff(filename: str, original: str, modified: str) -> str:
    """
    Build a minimal unified-diff that `git apply` can consume.
    `filename` is the path relative to the repo root.
    """
    import difflib
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=3,
    )
    return "".join(diff)


def _llm_propose_fix(issue: dict, source_code: str, filename: str) -> str | None:
    """
    Ask the LLM to rewrite the source file so the issue is fixed.
    Returns the FULL modified file content (NOT a diff) or None.
    Falls back to None if no LLM or invalid output.
    """
    if not _LLM_AVAILABLE or not os.getenv("GROQ_API_KEY"):
        return None

    try:
        llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_tokens=4500,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior C/C++ engineer fixing a single issue in an "
             "ESP-Matter firmware source file. Output ONLY the FULL modified "
             "file content — no markdown, no backticks, no commentary. "
             "Preserve every comment and the original coding style. "
             "Apply the smallest possible change that resolves the issue."),
            ("human",
             "File: {filename}\n"
             "Issue: {description}\n"
             "Suggested fix: {suggested_fix}\n\n"
             "=== ORIGINAL FILE ===\n{source}\n=== END ===\n\n"
             "Return ONLY the new full content of {filename}."),
        ])

        chain = prompt | llm | StrOutputParser()
        out = chain.invoke({
            "filename":      filename,
            "description":   issue.get("description", ""),
            "suggested_fix": issue.get("suggested_fix", ""),
            "source":        source_code,
        })
        out = out.strip()
        # Remove accidental fences if model added them
        out = re.sub(r"^```[a-zA-Z+]*\n", "", out)
        out = re.sub(r"\n```$", "", out)
        # Sanity check: must still look like C/C++
        if "{" in out and "}" in out and len(out) > 100:
            return out
    except Exception as e:
        print(f"[AutoFix] LLM call failed: {e}")
    return None


def _write_patch(name: str, diff: str) -> Path:
    PATCHES.mkdir(parents=True, exist_ok=True)
    p = PATCHES / name
    p.write_text(diff, encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════════════
# MAIN ENTRY-POINT — what the orchestrator calls
# ════════════════════════════════════════════════════════════════════
def run_autofix_agent(target: str = TARGET) -> dict:
    print(f"\n[AutoFix Agent] Starting for target: {target}")
    REPORTS.mkdir(exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)

    # 1. Load every agent report we care about ----------------------
    reports = {
        "security":   _load_json(REPORTS / f"security-report-{target}.json"),
        "code_review":_load_json(REPORTS / f"code-review-{target}.json"),
        "debug":      _load_json(REPORTS / f"debug-report-{target}.json"),
        "fault":      _load_json(REPORTS / f"fault-analysis-report-{target}.json"),
    }

    # 2. Find ESP-Matter source ------------------------------------
    src_dir = _find_source_dir()
    sources = _read_source(src_dir) if src_dir else {}
    print(f"[AutoFix] Source dir : {src_dir}")
    print(f"[AutoFix] Source files loaded: {list(sources.keys())}")

    # 3. Collect issues from all agents ----------------------------
    issues = _collect_issues(reports)
    print(f"[AutoFix] {len(issues)} total issue(s) collected")

    code_issues     = [i for i in issues if _is_code_issue(i)]
    non_code_issues = [i for i in issues if not _is_code_issue(i)]
    print(f"[AutoFix]  -> {len(code_issues)} code issue(s) "
          f"(will generate patches)")
    print(f"[AutoFix]  -> {len(non_code_issues)} non-code issue(s) "
          f"(instructions only)")

    # 4. Generate patches for code issues --------------------------
    patches_generated: list[dict] = []

    for idx, issue in enumerate(code_issues, start=1):
        # Pick which file to patch
        target_file = None
        for fn in SOURCE_FILES:
            if fn in (issue.get("file") or "") or fn in issue.get("description", ""):
                target_file = fn
                break
        if target_file is None and sources:
            # Fallback: patch app_main.cpp by default
            target_file = "app_main.cpp" if "app_main.cpp" in sources else next(iter(sources))

        if target_file is None or target_file not in sources:
            print(f"[AutoFix] Skip issue #{idx} — no source file to patch")
            continue

        original = sources[target_file]
        modified = _llm_propose_fix(issue, original, target_file)

        if not modified or modified == original:
            print(f"[AutoFix] Skip issue #{idx} — LLM did not produce a usable fix")
            continue

        rel_path = f"esp-matter/examples/light/main/{target_file}"
        diff = _make_unified_diff(rel_path, original, modified)
        if not diff.strip():
            continue

        patch_name = f"autofix-{target}-{idx:02d}-{issue['source_agent']}.patch"
        path = _write_patch(patch_name, diff)
        patches_generated.append({
            "name":         patch_name,
            "file":         rel_path,
            "source_agent": issue["source_agent"],
            "severity":     issue["severity"],
            "description":  issue["description"][:200],
        })
        # Update the in-memory source so subsequent issues stack
        sources[target_file] = modified
        print(f"[AutoFix] Patch generated: {patch_name}")

    # 5. Build instructions for non-code issues --------------------
    instructions: list[dict] = []
    for issue in non_code_issues:
        instructions.append({
            "source_agent": issue["source_agent"],
            "severity":     issue["severity"],
            "category":     issue.get("category", "config"),
            "description":  issue["description"],
            "how_to_fix":   issue.get("suggested_fix")
                            or "Manual review required — see description.",
        })

    # 6. Save consolidated report ----------------------------------
    report = {
        "agent":             "autofix",
        "target":            target,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "issues_analyzed":   len(issues),
        "patches_generated": len(patches_generated),
        "patches":           patches_generated,
        "manual_instructions": instructions,
        "summary": (
            f"AutoFix: {len(patches_generated)} patch(es) for source code, "
            f"{len(instructions)} manual instruction(s) for "
            f"config / CI / dependencies."
        ),
    }
    out = REPORTS / f"autofix-report-{target}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[AutoFix Agent] Report saved: {out}")
    print(f"[AutoFix Agent] Done — patches={len(patches_generated)}, "
          f"instructions={len(instructions)}")
    return report


# ════════════════════════════════════════════════════════════════════
# Standalone helpers (kept for backward compatibility with the
# original CLI entry-point).  The orchestrator uses run_autofix_agent.
# ════════════════════════════════════════════════════════════════════
def apply_patches() -> bool:
    """Apply every reports/patches/*.patch via `git apply` (legacy)."""
    patch_files = list(PATCHES.glob("*.patch"))
    if not patch_files:
        print("[AutoFix] No patches found")
        return False

    applied = False
    for pf in patch_files:
        print(f"[AutoFix] Applying {pf.name}")
        result = subprocess.run(
            ["git", "apply", str(pf)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  OK Applied {pf.name}")
            applied = True
        else:
            print(f"  Skipped {pf.name}")
    return applied


def run() -> dict:
    """Backward-compatible entry-point."""
    return run_autofix_agent()


if __name__ == "__main__":
    run_autofix_agent()
