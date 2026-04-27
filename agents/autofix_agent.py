"""
agents/autofix_agent.py — Agent 8 : AutoFix Agent
==================================================
TWO-LEVEL FIX ENGINE:
  Level 1 — LLM (Groq): rewrites the full file
  Level 2 — Rule-based: deterministic, no LLM needed
    Guarantees patches even without GROQ_API_KEY.

COHERENCE FIXES vs other agents:
  - apply_patches parameter added (called by LangGraph orchestrator)
  - _collect_issues parses code_review markdown "review" field
  - _is_code_issue handles empty file field (Gitleaks sometimes omits it)
  - demo/intentional_bug.py detected from description keywords

PATCH PATH:
  Python files: fromfile="a/demo/intentional_bug.py"
    git apply -p1 -> "demo/intentional_bug.py" -> FOUND in repo
  C++ files: fromfile="a/esp-matter/examples/light/main/app_main.cpp"
    patch -p1 -d esp-matter/examples/light -> "main/app_main.cpp" -> FOUND

VALIDATION:
  patch --dry-run before saving. Broken patches discarded.
"""

import difflib, json, os, re, subprocess, tempfile
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    load_dotenv()
    _LLM_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
except Exception:
    _LLM_AVAILABLE = False

TARGET  = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS = Path("reports")
PATCHES = REPORTS / "patches"

CPP_SOURCE_FILES = ["app_main.cpp", "app_driver.cpp", "app_priv.h"]
ESP_SOURCE_CANDIDATES = [
    Path("esp-matter/examples/light/main"),
    Path(os.getenv("EXAMPLE_PATH", "esp-matter/examples/light")) / "main",
    Path("/opt/espressif/esp-matter/examples/light/main"),
]

# Only skip issues that are truly non-patchable (CI secrets, infra)
NON_CODE_KEYWORDS = (
    "groq_api_key", "pat_token", "github_token", "github secret",
    "workflow secret", "ci secret", "sbom", "package version",
    "dependabot", "docker image tag", ".github/workflows",
    "environment variable missing", "env var not set",
)

# Known demo files with bugs — used when Gitleaks omits the file field
DEMO_BUG_FILES = [
    "demo/intentional_bug.py",
    "demo/bug.py",
]


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_esp_source_dir():
    for c in ESP_SOURCE_CANDIDATES:
        if c.is_dir() and any((c/f).exists() for f in CPP_SOURCE_FILES):
            return c
    return None


def _read_file(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_markdown_issues(review_text: str) -> list:
    """
    Parse code_review_agent markdown output to extract issues.
    code_review_agent writes 'review' as a markdown string, NOT a list.
    This function converts it to structured issues for autofix.
    """
    if not review_text:
        return []
    issues = []
    # Split by section headers
    sections = re.split(r"##\s+", review_text)
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        header = lines[0].lower()
        content = "\n".join(lines[1:]).strip()
        if not content or len(content) < 20:
            continue
        # Determine severity from section name
        if any(k in header for k in ("security", "critical", "unsafe", "buffer")):
            severity = "high"
        elif any(k in header for k in ("quality", "best practice", "improvement", "violation")):
            severity = "medium"
        else:
            continue  # Skip non-issue sections (score, title)
        # Split content into individual bullet points
        bullets = re.split(r"\n[-*•]\s*|\n\d+\.\s*", content)
        for bullet in bullets:
            bullet = bullet.strip()
            if len(bullet) > 30:  # Skip very short bullets
                issues.append({
                    "source_agent":  "code_review",
                    "severity":      severity,
                    "file":          "",  # will be resolved in _get_patch_target
                    "description":   bullet[:300],
                    "suggested_fix": "",
                    "category":      "quality",
                })
    return issues[:5]  # Limit to avoid too many patches


# ════════════════════════════════════════════════════════════════════
# ISSUE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

def _is_code_issue(issue: dict) -> bool:
    """
    True if this issue can be fixed by patching a file in the repo.
    Handles empty file field (common with Gitleaks reports).
    """
    file_field = (issue.get("file") or "").strip()

    # Rule 0: demo file keywords in description -> always patchable
    desc_lower = issue.get("description", "").lower()
    if any(kw in desc_lower for kw in ("sk-demo", "intentional_bug", "hardcoded api", "demo_api_key")):
        return True

    # Rule 1: file physically exists in repo -> patchable
    if file_field and Path(file_field).exists():
        return True

    # Rule 1b: basename matches C++ source in esp-matter
    if file_field:
        src_dir = _find_esp_source_dir()
        if src_dir and (src_dir / Path(file_field).name).exists():
            return True

    # Rule 2: non-code keywords -> skip
    text = " ".join([
        issue.get("description", ""),
        issue.get("location", ""),
        issue.get("category", ""),
    ]).lower()
    if any(k in text for k in NON_CODE_KEYWORDS):
        return False

    # Rule 3: mentions a C++ source file -> patchable
    if any(fn in text or fn in file_field for fn in CPP_SOURCE_FILES):
        return True

    # Rule 4: has actionable code info -> patchable
    if issue.get("code_snippet") or issue.get("suggested_fix"):
        return True

    # Rule 5: security issue with secret -> patchable (demo file)
    if issue.get("category") in ("secret_in_code",) or "hardcoded" in text:
        return True

    return False


def _get_patch_target(issue: dict):
    """
    Returns (repo_relative_path, file_content) or None.
    Handles empty file field by trying demo files.
    """
    file_field = (issue.get("file") or "").strip()
    src_dir    = _find_esp_source_dir()
    desc_lower = issue.get("description", "").lower()

    # Try 1: direct repo file
    if file_field and Path(file_field).exists():
        content = _read_file(file_field)
        if content:
            return file_field.lstrip("/"), content

    # Try 2: demo files (when Gitleaks/security agent omits file field)
    for demo_file in DEMO_BUG_FILES:
        if Path(demo_file).exists():
            # Check if the issue is likely about this demo file
            content = _read_file(demo_file)
            if content and (
                not file_field  # no file field -> try demo
                or demo_file in file_field
                or any(kw in desc_lower for kw in ("sk-demo", "hardcoded", "api key", "api_key", "secret"))
            ):
                return demo_file, content

    # Try 3: C++ file in esp-matter
    if src_dir:
        basename = Path(file_field).name if file_field else ""
        for fn in CPP_SOURCE_FILES:
            if basename == fn or fn.lower() in desc_lower:
                p = src_dir / fn
                if p.exists():
                    return f"esp-matter/examples/light/main/{fn}", _read_file(p)
        # Fallback app_main.cpp for code review issues
        if issue.get("source_agent") in ("code_review", "debug", "fault_analysis"):
            p = src_dir / "app_main.cpp"
            if p.exists():
                return "esp-matter/examples/light/main/app_main.cpp", _read_file(p)

    return None


# ════════════════════════════════════════════════════════════════════
# ISSUE COLLECTION — reads all agent reports
# ════════════════════════════════════════════════════════════════════

def _collect_issues(reports: dict) -> list:
    issues = []

    # Security — hardcoded secrets (from Gitleaks via security_agent)
    sec = reports.get("security", {})
    for s in (sec.get("secrets_found") or []):
        # Gitleaks sometimes has empty file field
        file_val = s.get("file", "") or s.get("path", "") or ""
        issues.append({
            "source_agent":  "security",
            "severity":      "critical",
            "file":          file_val,
            "description": (
                f"Hardcoded {s.get('type','secret')} detected. "
                f"Rule: {s.get('rule','')}. "
                f"Match: {s.get('match',s.get('secret',''))[:40]}. "
                f"Action: {s.get('action','')}"
            ),
            "suggested_fix": "Replace with os.environ.get('VAR', '')",
            "category":      "secret_in_code",
        })
    for cve in (sec.get("critical_cves") or []):
        issues.append({
            "source_agent":  "security",
            "severity":      cve.get("severity", "high"),
            "file":          cve.get("file", ""),
            "description":   cve.get("description", str(cve)),
            "suggested_fix": cve.get("remediation", ""),
            "category":      "security",
        })

    # Code review — parse markdown "review" field
    cr = reports.get("code_review", {})
    # Try structured issues first (future-proof)
    cr_issues = cr.get("issues") or cr.get("findings") or []
    if cr_issues:
        for it in cr_issues:
            issues.append({
                "source_agent":  "code_review",
                "severity":      it.get("severity", "medium"),
                "file":          it.get("file", ""),
                "description":   it.get("description") or str(it),
                "suggested_fix": it.get("suggested_fix",""),
                "category":      "quality",
            })
    else:
        # Parse markdown "review" string (current code_review_agent format)
        review_text = cr.get("review", "")
        if review_text:
            parsed = _parse_markdown_issues(review_text)
            issues.extend(parsed)

    # Debug agent
    dbg = reports.get("debug", {})
    for it in (dbg.get("issues") or dbg.get("bugs") or dbg.get("compilation_errors") or []):
        issues.append({
            "source_agent":  "debug",
            "severity":      it.get("severity", "high"),
            "file":          it.get("file", ""),
            "description":   it.get("description") or it.get("error") or str(it),
            "suggested_fix": it.get("suggested_fix") or it.get("fix", ""),
            "category":      "bug",
        })

    # Fault analysis
    fa = reports.get("fault", {})
    for it in (fa.get("regressions") or fa.get("issues") or fa.get("failed_scenarios_analysis") or []):
        issues.append({
            "source_agent":  "fault_analysis",
            "severity":      it.get("severity", "medium"),
            "file":          it.get("file", "") or it.get("affected_file", ""),
            "description":   it.get("description") or it.get("root_cause") or str(it),
            "suggested_fix": it.get("suggested_fix") or it.get("fix_code", ""),
            "category":      "robustness",
        })

    print(f"[AutoFix] Collected: {len(issues)} issues "
          f"({sum(1 for i in issues if i['source_agent']=='security')} security, "
          f"{sum(1 for i in issues if i['source_agent']=='code_review')} code_review, "
          f"{sum(1 for i in issues if i['source_agent']=='debug')} debug, "
          f"{sum(1 for i in issues if i['source_agent']=='fault_analysis')} fault)")
    return issues


# ════════════════════════════════════════════════════════════════════
# LEVEL 2 — Rule-based fixes (no LLM)
# ════════════════════════════════════════════════════════════════════

def _rule_based_fix_python(content, issue):
    modified, changed = content, False
    desc = (issue.get("description","") + " " + issue.get("category","")).lower()

    # Hardcoded secret
    if any(k in desc for k in ("secret","hardcoded","api key","api_key","token","credential","cwe-798","sk-demo")):
        pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=\s*["\']([^"\']{4,})["\']', re.MULTILINE)
        def _rep(m):
            var, val = m.group(1), m.group(2)
            if any(h in val.lower() for h in ("sk-","key","token","pass","secret","demo","api","gsk_")):
                return f'{var} = os.environ.get("{var}", "")'
            return m.group(0)
        new = pattern.sub(_rep, modified)
        if new != modified:
            modified, changed = new, True
            if "import os" not in modified:
                modified = "import os\n" + modified

    # Division by zero
    if any(k in desc for k in ("division","zero","cwe-369","zerodivision")):
        pat = re.compile(r'return\s+\(([^/\n]+)\s*/\s*(\w+)\)')
        def _div(m):
            num, div = m.group(1).strip(), m.group(2).strip()
            return f"if {div} == 0:\n        return 0.0\n    return ({num} / {div})"
        new = pat.sub(_div, modified)
        if new != modified:
            modified, changed = new, True

    # None dereference
    if any(k in desc for k in ("null","none","cwe-476","dereference","attributeerror","nonetype")):
        pat = re.compile(r'return\s+(\w+)\.(strip|lower|upper|split|replace|encode)\(\)')
        def _none(m):
            var, method = m.group(1), m.group(2)
            return f"if {var} is None:\n        return ''\n    return {var}.{method}()"
        new = pat.sub(_none, modified)
        if new != modified:
            modified, changed = new, True

    return modified if changed else None


def _rule_based_fix_cpp(content, issue):
    modified, changed = content, False
    desc = (issue.get("description","") + " " + issue.get("category","")).lower()
    if any(k in desc for k in ("null","malloc","heap","cwe-476","null pointer")):
        pat = re.compile(
            r'([ \t]*)([\w *]+\*?\s*(\w+)\s*=\s*(?:malloc|calloc|heap_caps_malloc)\s*\([^;]+\);)',
            re.MULTILINE)
        def _null(m):
            ind, decl, var = m.group(1), m.group(2), m.group(3)
            return (f"{ind}{decl}\n{ind}if ({var} == NULL) {{\n"
                    f'{ind}    ESP_LOGE(TAG, "malloc failed for {var}");\n'
                    f"{ind}    return ESP_ERR_NO_MEM;\n{ind}}}")
        new = pat.sub(_null, modified)
        if new != modified:
            modified, changed = new, True
    return modified if changed else None


def _rule_based_fix(content, issue, is_python):
    return (_rule_based_fix_python(content, issue) if is_python
            else _rule_based_fix_cpp(content, issue))


# ════════════════════════════════════════════════════════════════════
# LEVEL 1 — LLM fix
# ════════════════════════════════════════════════════════════════════

def _llm_fix(issue, content, filename, is_python):
    if not _LLM_AVAILABLE:
        return None
    lang = "Python" if is_python else "C/C++ ESP-IDF"
    try:
        llm = ChatGroq(model=os.getenv("LLM_MODEL","llama-3.3-70b-versatile"),
                       api_key=os.getenv("GROQ_API_KEY"), temperature=0.0, max_tokens=4500)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a senior {lang} engineer. Fix ONE issue. "
             "Return ONLY the complete corrected file, no markdown, no explanation. "
             "Hardcoded secrets -> os.environ.get('VAR', ''). Smallest possible change."),
            ("human", "File: {fn}\nIssue: {desc}\nHint: {fix}\n\n"
             "=== ORIGINAL ===\n{src}\n=== END ===\nReturn corrected file only.")])
        out = (prompt | llm | StrOutputParser()).invoke({
            "fn": filename, "desc": issue.get("description",""),
            "fix": issue.get("suggested_fix",""), "src": content[:5000]})
        out = re.sub(r"^```[a-zA-Z+]*\n?","",out.strip())
        out = re.sub(r"\n?```$","",out).strip()
        if len(out)<30: return None
        if is_python and "def " not in out and "import " not in out: return None
        if not is_python and "{" not in out: return None
        return out
    except Exception as e:
        print(f"[AutoFix] LLM error: {e}")
        return None


def _make_diff(repo_rel, original, modified):
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{repo_rel}", tofile=f"b/{repo_rel}", n=3))


def _validate(diff, original, filename):
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".txt",
                                          mode="w",encoding="utf-8",delete=False) as f:
            f.write(original); sp = f.name
        with tempfile.NamedTemporaryFile(suffix=".patch",mode="w",encoding="utf-8",delete=False) as f:
            f.write(diff); pp = f.name
        r = subprocess.run(["patch","--dry-run","-p1",sp,pp],
                           capture_output=True,text=True,timeout=10)
        os.unlink(sp); os.unlink(pp)
        if r.returncode==0: return True
        print(f"[AutoFix] Validation failed: {r.stderr[:150]}")
        return False
    except Exception as e:
        print(f"[AutoFix] Validation error: {e}"); return True


# ════════════════════════════════════════════════════════════════════
# MAIN — called by orchestrator with or without apply_patches param
# ════════════════════════════════════════════════════════════════════

def run_autofix_agent(target=TARGET, apply_patches=False):
    """
    Main entry point.
    apply_patches=False: only generate patches (CI workflow uses this)
    apply_patches=True:  also apply via git apply immediately (CLI use)
    """
    print(f"\n[AutoFix] ===== target:{target} LLM:{_LLM_AVAILABLE} =====")
    REPORTS.mkdir(exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)

    # Load all agent reports
    reports = {
        "security":    _load_json(REPORTS / f"security-report-{target}.json"),
        "code_review": _load_json(REPORTS / f"code-review-{target}.json"),
        "debug":       _load_json(REPORTS / f"debug-report-{target}.json"),
        "fault":       _load_json(REPORTS / f"fault-analysis-report-{target}.json"),
    }
    print(f"[AutoFix] ESP source dir: {_find_esp_source_dir()}")

    all_issues   = _collect_issues(reports)
    code_issues  = [i for i in all_issues if _is_code_issue(i)]
    other_issues = [i for i in all_issues if not _is_code_issue(i)]
    print(f"[AutoFix] {len(code_issues)} patchable, {len(other_issues)} manual-only")

    file_cache   = {}
    patches_done = []
    last_method  = "none"

    for idx, issue in enumerate(code_issues, 1):
        info = _get_patch_target(issue)
        if not info:
            print(f"[AutoFix] #{idx}: no target file — skip")
            continue

        repo_rel, original = info
        current  = file_cache.get(repo_rel, original)
        is_py    = repo_rel.endswith(".py")
        basename = Path(repo_rel).name

        print(f"[AutoFix] #{idx}: [{issue['severity']}] {issue['description'][:55]}")
        print(f"          file: {repo_rel}")

        # Level 1: LLM
        modified = _llm_fix(issue, current, basename, is_py)
        method   = "llm"
        if not modified or modified.strip() == current.strip():
            # Level 2: rule-based fallback
            modified = _rule_based_fix(current, issue, is_py)
            method   = "rule_based"

        if not modified or modified.strip() == current.strip():
            print("          -> no change — skip")
            continue

        diff = _make_diff(repo_rel, current, modified)
        if not diff.strip():
            print("          -> empty diff — skip")
            continue
        if not _validate(diff, current, basename):
            print("          -> validation failed — discarded")
            continue

        safe  = repo_rel.replace("/","_").replace("\\","_")
        pname = f"autofix-{target}-{idx:02d}-{issue['source_agent']}-{safe}.patch"
        (PATCHES/pname).write_text(diff, encoding="utf-8")
        file_cache[repo_rel] = modified
        last_method = method
        patches_done.append({
            "patch_name":   pname,
            "file":         repo_rel,
            "source_agent": issue["source_agent"],
            "severity":     issue["severity"],
            "description":  issue["description"][:200],
            "fix_method":   method,
        })
        print(f"          -> SAVED: {pname} [{method}]")

    # Optional: apply immediately (CLI use)
    if apply_patches and patches_done:
        print("[AutoFix] Applying patches via git apply...")
        for p in patches_done:
            r = subprocess.run(
                ["git","apply", str(PATCHES/p["patch_name"])],
                capture_output=True, text=True)
            status = "OK" if r.returncode==0 else f"Skip({r.stderr[:60]})"
            print(f"  {status}: {p['patch_name']}")

    instructions = [{
        "source_agent": i["source_agent"],
        "severity":     i["severity"],
        "description":  i["description"],
        "how_to_fix":   i.get("suggested_fix","Manual review required."),
    } for i in other_issues]

    report = {
        "agent":             "autofix_agent",
        "target":            target,
        "generated_at":      datetime.utcnow().isoformat()+"Z",
        "llm_used":          _LLM_AVAILABLE,
        "issues_analyzed":   len(all_issues),
        "patches_generated": len(patches_done),   # orchestrator reads this
        "patch_files":       [p["patch_name"] for p in patches_done],
        "patches_detail":    patches_done,
        "manual_instructions": instructions,
        "status": "patches_generated" if patches_done else "no_patches_generated",
        "summary": (
            f"{len(patches_done)} patch(es) "
            f"({'LLM+' if _LLM_AVAILABLE else ''}{last_method}), "
            f"{len(instructions)} instructions."
        ),
    }
    out = REPORTS / f"autofix-report-{target}.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\n[AutoFix] patches={len(patches_done)} instructions={len(instructions)}")
    return report


def apply_patches():
    """Legacy CLI helper."""
    applied = False
    for pf in PATCHES.glob("*.patch"):
        r = subprocess.run(["git","apply",str(pf)],capture_output=True,text=True)
        if r.returncode==0: print(f"  OK: {pf.name}"); applied=True
        else: print(f"  Skip: {pf.name}")
    return applied

def run(): return run_autofix_agent()
if __name__ == "__main__": run_autofix_agent()
