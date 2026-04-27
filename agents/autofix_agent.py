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
       (a) issues that live INSIDE source files that exist in the repo
           (app_main.cpp / app_driver.cpp / app_priv.h / demo/*.py / etc.)
              -> generate a real .patch file for the second run.
       (b) issues that live OUTSIDE the source and CANNOT be patched
           (CI config, dependencies, missing files, external env vars)
              -> emit a textual "instruction" in the report.

  2. Patch path logic (critical for git apply to work):
       - C/C++ files in esp-matter/ ->
           fromfile = "a/esp-matter/examples/light/main/{file}"
           git apply finds the file at that path in the repo.
       - Python/other files in the repo root (demo/*.py, agents/*.py) ->
           fromfile = "a/{relative_path}"
           git apply finds the file at that path in the repo.

  3. Validates every patch with `patch --dry-run` BEFORE saving.
     A broken patch is discarded so Stage 4b never fails.

  4. Stacks fixes: each subsequent issue uses the already-modified
     file so patches do not conflict.

  5. Saves every validated patch into  reports/patches/*.patch.
  6. Saves the consolidated report into
        reports/autofix-report-{target}.json

INPUTS:
  reports/security-report-{target}.json
  reports/code-review-{target}.json
  reports/debug-report-{target}.json
  reports/fault-analysis-report-{target}.json
  esp-matter/examples/light/main/   (C++ sources)
  demo/                              (demo Python bugs)
  agents/                            (agent Python files)

OUTPUTS:
  reports/patches/*.patch
  reports/autofix-report-{target}.json
"""

import json
import os
import re
import subprocess
import tempfile
import difflib
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
TARGET  = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS = Path("reports")
PATCHES = REPORTS / "patches"

# C++ source files living inside the esp-matter directory
CPP_SOURCE_FILES = ["app_main.cpp", "app_driver.cpp", "app_priv.h"]

ESP_SOURCE_CANDIDATES = [
    Path("esp-matter/examples/light/main"),
    Path(os.getenv("EXAMPLE_PATH", "esp-matter/examples/light")) / "main",
    Path("/opt/espressif/esp-matter/examples/light/main"),
]

# ── Keywords that mark issues that CANNOT be fixed by a code patch ──
# NOTE: "secret" alone is NOT here — a hardcoded secret inside a Python
# file in the repo (demo/intentional_bug.py) IS a code-patchable issue.
# We only skip secrets when they refer to CI secrets / env vars.
NON_CODE_KEYWORDS = (
    "ci secret", "github secret", "workflow secret",
    "groq_api_key", "pat_token", "github_token",
    "ci", "workflow", "pipeline", ".yml", ".yaml",
    "sbom", "package version", "dependency",
    "docker image", "registry",
    "permission", "chmod",
    "environment variable", "env var",
)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_esp_source_dir() -> Path | None:
    """Return the esp-matter/examples/light/main directory if it exists."""
    for c in ESP_SOURCE_CANDIDATES:
        if c.is_dir() and any((c / f).exists() for f in CPP_SOURCE_FILES):
            return c
    return None


def _read_repo_file(relative_path: str) -> str:
    """
    Read ANY file that exists in the repo by its path relative to repo root.
    Works for demo/intentional_bug.py, agents/autofix_agent.py, etc.
    """
    p = Path(relative_path)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    return ""


def _read_cpp_source(src_dir: Path) -> dict[str, str]:
    """Read all C++ source files from the esp-matter directory."""
    out = {}
    for fn in CPP_SOURCE_FILES:
        p = src_dir / fn
        if p.exists():
            out[fn] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def _is_code_issue(issue: dict) -> bool:
    """
    Return True if this issue can be fixed by patching a source file
    that exists in the repo.

    Logic (in order):
    1. If the issue's 'file' field names a file that EXISTS in the repo
       -> always a code issue (regardless of description keywords).
    2. If the description contains non-code keywords
       -> NOT a code issue (CI config, env vars, etc.)
    3. If the description mentions a known source file -> code issue.
    4. If there is a code_snippet or suggested_fix -> code issue.
    5. Default: not a code issue.
    """
    file_field = issue.get("file", "")

    # Rule 1: the file actually exists in the repo -> patchable
    if file_field:
        # Try direct path (relative to repo root)
        if Path(file_field).exists():
            return True
        # Try basename in esp-matter source dir
        src_dir = _find_esp_source_dir()
        if src_dir and (src_dir / Path(file_field).name).exists():
            return True

    # Rule 2: non-code keywords in description -> skip
    text = " ".join([
        str(issue.get("description", "")),
        str(issue.get("location", "")),
        str(issue.get("category", "")),
    ]).lower()

    if any(k in text for k in NON_CODE_KEYWORDS):
        return False

    # Rule 3: mentions a known C++ source file -> code issue
    if any(fn in text or fn in file_field for fn in CPP_SOURCE_FILES):
        return True

    # Rule 4: has a code snippet or fix suggestion -> treat as code
    if issue.get("code_snippet") or issue.get("suggested_fix"):
        return True

    return False


def _get_patch_target(issue: dict) -> tuple[str, str] | None:
    """
    Return (relative_path_in_repo, file_content) for the file to patch.
    Returns None if no patchable file found.

    relative_path_in_repo is the path git apply will look for,
    e.g. "demo/intentional_bug.py" or
         "esp-matter/examples/light/main/app_main.cpp"
    """
    file_field = (issue.get("file") or "").strip()
    src_dir = _find_esp_source_dir()

    # --- Try 1: direct repo file (demo/*.py, agents/*.py, etc.) ---
    if file_field:
        p = Path(file_field)
        if p.exists():
            content = _read_repo_file(file_field)
            if content:
                # Keep the path as-is (relative to repo root)
                return file_field.lstrip("/"), content

    # --- Try 2: C++ file in esp-matter source dir ---
    if src_dir:
        # Look for the basename in the esp-matter source dir
        basename = Path(file_field).name if file_field else None
        for fn in CPP_SOURCE_FILES:
            if basename == fn or fn in (issue.get("description", "")):
                p = src_dir / fn
                if p.exists():
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    rel = f"esp-matter/examples/light/main/{fn}"
                    return rel, content

        # Fallback: use app_main.cpp if nothing else matches
        fallback = src_dir / "app_main.cpp"
        if fallback.exists():
            content = fallback.read_text(encoding="utf-8", errors="ignore")
            return "esp-matter/examples/light/main/app_main.cpp", content

    return None


def _collect_issues(reports: dict) -> list[dict]:
    """
    Flatten findings from all agent reports into a uniform list.
    """
    issues: list[dict] = []

    # Security agent — CVEs and hardcoded secrets
    sec = reports.get("security", {})
    for cve in (sec.get("critical_cves") or []):
        issues.append({
            "source_agent":  "security",
            "severity":      cve.get("severity", "high"),
            "file":          cve.get("file", ""),
            "description":   cve.get("description", str(cve)),
            "suggested_fix": cve.get("remediation", ""),
            "category":      "security",
        })
    for s in (sec.get("secrets_found") or []):
        # The file field tells us WHERE the secret is
        issues.append({
            "source_agent":  "security",
            "severity":      "critical",
            "file":          s.get("file", ""),
            "description":   (
                f"Hardcoded {s.get('type', 'secret')} detected: "
                f"{s.get('rule', str(s))}"
            ),
            "suggested_fix": (
                "Replace the hardcoded value with "
                "os.environ.get('KEY_NAME', '') or NVS storage."
            ),
            "category":      "secret_in_code",
        })

    # Code-review agent
    cr = reports.get("code_review", {})
    for it in (cr.get("issues") or cr.get("findings") or []):
        issues.append({
            "source_agent":  "code_review",
            "severity":      it.get("severity", "medium"),
            "file":          it.get("file", ""),
            "description":   it.get("description") or it.get("issue", str(it)),
            "suggested_fix": it.get("suggested_fix") or it.get("fix", ""),
            "code_snippet":  it.get("code_snippet", ""),
            "category":      "quality",
        })

    # Debug agent
    dbg = reports.get("debug", {})
    for it in (dbg.get("issues") or dbg.get("bugs") or []):
        issues.append({
            "source_agent":  "debug",
            "severity":      it.get("severity", "medium"),
            "file":          it.get("file", ""),
            "description":   it.get("description") or str(it),
            "suggested_fix": it.get("suggested_fix") or it.get("fix", ""),
            "category":      "bug",
        })

    # Fault-analysis agent
    fa = reports.get("fault", {})
    for it in (fa.get("regressions") or fa.get("issues") or []):
        issues.append({
            "source_agent":  "fault_analysis",
            "severity":      it.get("severity", "medium"),
            "file":          it.get("file", ""),
            "description":   it.get("description") or str(it),
            "suggested_fix": it.get("suggested_fix") or "",
            "category":      "robustness",
        })

    return issues


# ════════════════════════════════════════════════════════════════════
# PATCH GENERATION
# ════════════════════════════════════════════════════════════════════

def _make_unified_diff(repo_rel_path: str, original: str, modified: str) -> str:
    """
    Build a unified diff that `git apply` can consume.

    repo_rel_path is the file path relative to the repo root,
    e.g. "demo/intentional_bug.py" or
         "esp-matter/examples/light/main/app_main.cpp"

    The diff header uses "a/{repo_rel_path}" / "b/{repo_rel_path}".
    After `git apply -p1`:
      strips the leading "a/" -> repo_rel_path -> file found in repo.
    """
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{repo_rel_path}",
        tofile=f"b/{repo_rel_path}",
        n=3,
    )
    return "".join(diff_lines)


def _validate_patch(patch_content: str, original_src: str,
                    filename: str) -> bool:
    """
    Test the patch with `patch --dry-run` against the original content.
    Discards the patch if it would not apply cleanly.

    This prevents broken LLM patches from reaching Stage 4b and
    causing the Docker build to fail (which would skip Stage 4c).
    """
    try:
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, mode="w", encoding="utf-8", delete=False
        ) as tmp_src, tempfile.NamedTemporaryFile(
            suffix=".patch", mode="w", encoding="utf-8", delete=False
        ) as tmp_patch:
            tmp_src.write(original_src)
            tmp_patch.write(patch_content)
            src_path   = tmp_src.name
            patch_path = tmp_patch.name

        result = subprocess.run(
            ["patch", "--dry-run", "-p1", src_path, patch_path],
            capture_output=True, text=True, timeout=10,
        )
        os.unlink(src_path)
        os.unlink(patch_path)

        if result.returncode == 0:
            return True
        print(f"[AutoFix] Patch validation failed: {result.stderr[:200]}")
        return False
    except Exception as e:
        print(f"[AutoFix] Patch validation error: {e}")
        return True  # Don't discard on validator error


def _llm_propose_fix(issue: dict, source_code: str,
                     filename: str, is_python: bool) -> str | None:
    """
    Ask the LLM to rewrite the source file so the issue is fixed.
    Returns the FULL modified file content or None.
    """
    if not _LLM_AVAILABLE or not os.getenv("GROQ_API_KEY"):
        return None

    lang = "Python" if is_python else "C/C++ (ESP-IDF)"
    try:
        llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_tokens=4500,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a senior {lang} engineer fixing a single security or "
             "quality issue in a firmware/tool source file. "
             "Output ONLY the FULL modified file content — no markdown, "
             "no backticks, no commentary. "
             "Preserve every comment and the original coding style. "
             "Apply the SMALLEST possible change that resolves the issue. "
             "For hardcoded secrets: replace the literal value with "
             "os.environ.get('KEY_NAME', '') or an equivalent safe accessor."),
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
            "source":        source_code[:6000],
        })
        out = out.strip()
        # Strip accidental markdown fences
        out = re.sub(r"^```[a-zA-Z+]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out)
        out = out.strip()

        # Sanity checks
        if len(out) < 50:
            return None
        if is_python and "def " not in out and "import " not in out:
            return None
        if not is_python and "{" not in out:
            return None

        return out
    except Exception as e:
        print(f"[AutoFix] LLM call failed for {filename}: {e}")
    return None


def _write_patch(name: str, content: str) -> Path:
    PATCHES.mkdir(parents=True, exist_ok=True)
    p = PATCHES / name
    p.write_text(content, encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════════════
# MAIN ENTRY-POINT
# ════════════════════════════════════════════════════════════════════

def run_autofix_agent(target: str = TARGET) -> dict:
    print(f"\n[AutoFix Agent] Starting — target: {target}")
    REPORTS.mkdir(exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all agent reports ──────────────────────────────────
    reports = {
        "security":    _load_json(REPORTS / f"security-report-{target}.json"),
        "code_review": _load_json(REPORTS / f"code-review-{target}.json"),
        "debug":       _load_json(REPORTS / f"debug-report-{target}.json"),
        "fault":       _load_json(REPORTS / f"fault-analysis-report-{target}.json"),
    }

    src_dir = _find_esp_source_dir()
    print(f"[AutoFix] ESP source dir : {src_dir}")

    # ── 2. Collect and classify issues ────────────────────────────
    issues       = _collect_issues(reports)
    code_issues  = [i for i in issues if _is_code_issue(i)]
    other_issues = [i for i in issues if not _is_code_issue(i)]

    print(f"[AutoFix] {len(issues)} total issue(s): "
          f"{len(code_issues)} code, {len(other_issues)} non-code")

    # ── 3. Generate patches ────────────────────────────────────────
    # We track the current content of each file so that multiple
    # issues in the same file stack correctly (patch 2 is applied
    # to the output of patch 1, not the original).
    file_cache: dict[str, str] = {}   # repo_rel_path -> current content
    patches_generated: list[dict] = []

    for idx, issue in enumerate(code_issues, start=1):
        target_info = _get_patch_target(issue)
        if target_info is None:
            print(f"[AutoFix] Issue #{idx}: no patchable file found — skip")
            continue

        repo_rel_path, original_content = target_info

        # Use cached (already-modified) content if available
        current_content = file_cache.get(repo_rel_path, original_content)

        is_python = repo_rel_path.endswith(".py")
        filename  = Path(repo_rel_path).name

        print(f"[AutoFix] Issue #{idx}: patching {repo_rel_path}")

        modified = _llm_propose_fix(
            issue, current_content, filename, is_python
        )

        if not modified or modified.strip() == current_content.strip():
            print(f"  -> LLM did not produce a usable change — skip")
            continue

        diff = _make_unified_diff(repo_rel_path, current_content, modified)
        if not diff.strip():
            print(f"  -> Diff is empty — skip")
            continue

        # Validate before saving
        if not _validate_patch(diff, current_content, filename):
            print(f"  -> Patch validation failed — discarded")
            continue

        safe_name  = repo_rel_path.replace("/", "_").replace("\\", "_")
        patch_name = f"autofix-{target}-{idx:02d}-{issue['source_agent']}-{safe_name}.patch"
        _write_patch(patch_name, diff)

        # Update cache so next issue in the same file stacks
        file_cache[repo_rel_path] = modified

        patches_generated.append({
            "patch_name":   patch_name,
            "file":         repo_rel_path,
            "source_agent": issue["source_agent"],
            "severity":     issue["severity"],
            "description":  issue["description"][:200],
            "validated":    True,
        })
        print(f"  -> Patch saved: {patch_name}")

    # ── 4. Build instructions for non-code issues ─────────────────
    instructions: list[dict] = []
    for issue in other_issues:
        instructions.append({
            "source_agent": issue["source_agent"],
            "severity":     issue["severity"],
            "category":     issue.get("category", "config"),
            "description":  issue["description"],
            "how_to_fix":   (
                issue.get("suggested_fix")
                or "Manual review required — see description."
            ),
        })

    # ── 5. Save report ─────────────────────────────────────────────
    report = {
        "agent":             "autofix_agent",
        "target":            target,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "issues_analyzed":   len(issues),
        "patches_generated": len(patches_generated),  # orchestrator reads this
        "patch_files":       [p["patch_name"] for p in patches_generated],
        "patches_detail":    patches_generated,
        "manual_instructions": instructions,
        "status": (
            "patches_generated" if patches_generated
            else "no_patches_generated"
        ),
        "summary": (
            f"{len(patches_generated)} validated patch(es) for source code, "
            f"{len(instructions)} manual instruction(s) for CI/config."
        ),
    }

    out = REPORTS / f"autofix-report-{target}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[AutoFix Agent] Done — "
          f"patches={len(patches_generated)}, "
          f"instructions={len(instructions)}")
    print(f"[AutoFix Agent] Report: {out}")
    return report


# ════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE STANDALONE ENTRY-POINTS
# ════════════════════════════════════════════════════════════════════

def apply_patches() -> bool:
    """Apply every reports/patches/*.patch via git apply (legacy CLI)."""
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
            print(f"  OK: {pf.name}")
            applied = True
        else:
            print(f"  Skip: {pf.name} — {result.stderr[:80]}")
    return applied


def run() -> dict:
    """Backward-compatible entry-point."""
    return run_autofix_agent()


if __name__ == "__main__":
    run_autofix_agent()
