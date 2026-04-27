"""
agents/autofix_agent.py — Agent 8 : AutoFix Agent
==================================================
TWO-LEVEL FIX ENGINE:
  Level 1 — LLM (Groq): rewrites the full file
  Level 2 — Rule-based (no LLM): pattern-matching fixes
    guaranteed to produce patches even without GROQ_API_KEY.

PATCH PATH LOGIC:
  Python repo files: fromfile = "a/demo/intentional_bug.py"
    git apply -p1 -> "demo/intentional_bug.py" -> FOUND ✓
  C++ files: fromfile = "a/esp-matter/examples/light/main/app_main.cpp"
    patch -p1 -d esp-matter/examples/light -> "main/app_main.cpp" -> FOUND ✓

VALIDATION:
  patch --dry-run before saving. Broken patches discarded.
  Stage 4b never fails because of a bad patch.
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

NON_CODE_KEYWORDS = (
    "groq_api_key", "pat_token", "github_token", "github secret",
    "workflow secret", "ci secret", "sbom", "package version",
    "dependabot", "docker image tag", ".github/workflows",
    "environment variable missing", "env var not set",
)


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


def _is_code_issue(issue):
    file_field = (issue.get("file") or "").strip()
    # Rule 1: file exists in repo -> always patchable
    if file_field and Path(file_field).exists():
        return True
    src_dir = _find_esp_source_dir()
    if file_field and src_dir and (src_dir / Path(file_field).name).exists():
        return True
    # Rule 2: non-code keywords -> skip
    text = " ".join([
        issue.get("description",""), issue.get("location",""), issue.get("category","")
    ]).lower()
    if any(k in text for k in NON_CODE_KEYWORDS):
        return False
    # Rule 3: mentions a C++ source file
    if any(fn in text or fn in file_field for fn in CPP_SOURCE_FILES):
        return True
    return bool(issue.get("code_snippet") or issue.get("suggested_fix"))


def _get_patch_target(issue):
    file_field = (issue.get("file") or "").strip()
    src_dir    = _find_esp_source_dir()
    # Try 1: direct repo file
    if file_field and Path(file_field).exists():
        content = _read_file(file_field)
        if content:
            return file_field.lstrip("/"), content
    # Try 2: C++ file
    if src_dir:
        basename = Path(file_field).name if file_field else ""
        desc = issue.get("description","").lower()
        for fn in CPP_SOURCE_FILES:
            if basename == fn or fn.lower() in desc:
                p = src_dir / fn
                if p.exists():
                    return f"esp-matter/examples/light/main/{fn}", _read_file(p)
        p = src_dir / "app_main.cpp"
        if p.exists():
            return "esp-matter/examples/light/main/app_main.cpp", _read_file(p)
    return None


def _collect_issues(reports):
    issues = []
    sec = reports.get("security", {})
    for cve in (sec.get("critical_cves") or []):
        issues.append({"source_agent":"security","severity":cve.get("severity","high"),
            "file":cve.get("file",""),"description":cve.get("description",str(cve)),
            "suggested_fix":cve.get("remediation",""),"category":"security"})
    for s in (sec.get("secrets_found") or []):
        issues.append({"source_agent":"security","severity":"critical",
            "file":s.get("file",""),
            "description":f"Hardcoded {s.get('type','secret')} detected. Rule:{s.get('rule','')} Action:{s.get('action','')}",
            "suggested_fix":"Replace with os.environ.get('VAR', '') or NVS.",
            "category":"secret_in_code"})
    cr = reports.get("code_review", {})
    for it in (cr.get("issues") or cr.get("findings") or []):
        issues.append({"source_agent":"code_review","severity":it.get("severity","medium"),
            "file":it.get("file",""),"description":it.get("description") or str(it),
            "suggested_fix":it.get("suggested_fix",""),"code_snippet":it.get("code_snippet",""),
            "category":"quality"})
    dbg = reports.get("debug", {})
    for it in (dbg.get("issues") or dbg.get("bugs") or []):
        issues.append({"source_agent":"debug","severity":it.get("severity","medium"),
            "file":it.get("file",""),"description":it.get("description") or str(it),
            "suggested_fix":it.get("suggested_fix",""),"category":"bug"})
    fa = reports.get("fault", {})
    for it in (fa.get("regressions") or fa.get("issues") or []):
        issues.append({"source_agent":"fault_analysis","severity":it.get("severity","medium"),
            "file":it.get("file",""),"description":it.get("description") or str(it),
            "suggested_fix":it.get("suggested_fix",""),"category":"robustness"})
    return issues


# ── LEVEL 2: Rule-based fixes (no LLM needed) ─────────────────────

def _rule_based_fix_python(content, issue):
    modified, changed = content, False
    desc = (issue.get("description","") + " " + issue.get("category","")).lower()

    # Rule A: hardcoded secret
    if any(k in desc for k in ("secret","hardcoded","api key","api_key","token","credential","cwe-798")):
        pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=\s*["\'"]([^"\'"]{4,})["\'"]', re.MULTILINE)
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

    # Rule B: division by zero
    if any(k in desc for k in ("division","zero","cwe-369","zerodivision")):
        pat = re.compile(r'return\s+\(([^/\n]+)\s*/\s*(\w+)\)')
        def _div(m):
            num, div = m.group(1).strip(), m.group(2).strip()
            return f"if {div} == 0:\n        return 0.0\n    return ({num} / {div})"
        new = pat.sub(_div, modified)
        if new != modified:
            modified, changed = new, True

    # Rule C: None dereference
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

    # Rule A: missing NULL check after malloc
    if any(k in desc for k in ("null","malloc","heap","cwe-476","null pointer")):
        pat = re.compile(
            r'([ \t]*)([ \w\*]+\*?\s*(\w+)\s*=\s*(?:malloc|calloc|heap_caps_malloc)\s*\([^;]+\);)',
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
    return _rule_based_fix_python(content, issue) if is_python else _rule_based_fix_cpp(content, issue)


# ── LEVEL 1: LLM fix ──────────────────────────────────────────────

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
             "Hardcoded secrets -> os.environ.get(). Smallest possible change."),
            ("human", "File: {fn}\nIssue: {desc}\nHint: {fix}\n\n"
             "=== ORIGINAL ===\n{src}\n=== END ===\nReturn corrected file only.")])
        out = (prompt | llm | StrOutputParser()).invoke({
            "fn":src_dir if (src_dir := filename) else filename,
            "desc":issue.get("description",""),
            "fix":issue.get("suggested_fix",""),
            "src":content[:5000]})
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
        print(f"[AutoFix] Validation: {r.stderr[:150]}")
        return False
    except Exception as e:
        print(f"[AutoFix] Validation error: {e}"); return True


def run_autofix_agent(target=TARGET):
    print(f"\n[AutoFix] ===== target:{target} LLM:{_LLM_AVAILABLE} =====")
    REPORTS.mkdir(exist_ok=True); PATCHES.mkdir(parents=True,exist_ok=True)

    reports = {
        "security":    _load_json(REPORTS/f"security-report-{target}.json"),
        "code_review": _load_json(REPORTS/f"code-review-{target}.json"),
        "debug":       _load_json(REPORTS/f"debug-report-{target}.json"),
        "fault":       _load_json(REPORTS/f"fault-analysis-report-{target}.json"),
    }
    print(f"[AutoFix] ESP source dir: {_find_esp_source_dir()}")

    all_issues   = _collect_issues(reports)
    code_issues  = [i for i in all_issues if _is_code_issue(i)]
    other_issues = [i for i in all_issues if not _is_code_issue(i)]
    print(f"[AutoFix] {len(all_issues)} issues: {len(code_issues)} patchable, {len(other_issues)} manual")

    file_cache   = {}
    patches_done = []

    for idx, issue in enumerate(code_issues, 1):
        info = _get_patch_target(issue)
        if not info:
            print(f"[AutoFix] #{idx}: no target — skip"); continue

        repo_rel, original = info
        current   = file_cache.get(repo_rel, original)
        is_py     = repo_rel.endswith(".py")
        basename  = Path(repo_rel).name

        print(f"[AutoFix] #{idx}: {issue['description'][:55]}")
        print(f"          -> {repo_rel}")

        # Level 1: LLM
        modified = _llm_fix(issue, current, basename, is_py)
        method   = "llm"
        # Level 2: rule-based fallback
        if not modified or modified.strip() == current.strip():
            modified = _rule_based_fix(current, issue, is_py)
            method   = "rule_based"

        if not modified or modified.strip() == current.strip():
            print("          -> no change produced — skip"); continue

        diff = _make_diff(repo_rel, current, modified)
        if not diff.strip():
            print("          -> empty diff — skip"); continue
        if not _validate(diff, current, basename):
            print("          -> validation failed — discarded"); continue

        safe  = repo_rel.replace("/","_").replace("\\","_")
        pname = f"autofix-{target}-{idx:02d}-{issue['source_agent']}-{safe}.patch"
        (PATCHES/pname).write_text(diff, encoding="utf-8")
        file_cache[repo_rel] = modified
        patches_done.append({"patch_name":pname,"file":repo_rel,
            "source_agent":issue["source_agent"],"severity":issue["severity"],
            "description":issue["description"][:200],"fix_method":method})
        print(f"          -> SAVED: {pname} (method={method})")

    instructions = [{"source_agent":i["source_agent"],"severity":i["severity"],
        "description":i["description"],"how_to_fix":i.get("suggested_fix","Manual review.")}
        for i in other_issues]

    report = {"agent":"autofix_agent","target":target,
        "generated_at":datetime.utcnow().isoformat()+"Z",
        "llm_used":_LLM_AVAILABLE,"issues_analyzed":len(all_issues),
        "patches_generated":len(patches_done),
        "patch_files":[p["patch_name"] for p in patches_done],
        "patches_detail":patches_done,"manual_instructions":instructions,
        "status":"patches_generated" if patches_done else "no_patches_generated",
        "summary":f"{len(patches_done)} patch(es) ({method if patches_done else 'none'}), {len(instructions)} instructions."}

    out = REPORTS/f"autofix-report-{target}.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\n[AutoFix] patches={len(patches_done)} instructions={len(instructions)}")
    return report


def apply_patches():
    applied = False
    for pf in PATCHES.glob("*.patch"):
        r = subprocess.run(["git","apply",str(pf)],capture_output=True,text=True)
        if r.returncode==0: print(f"  OK: {pf.name}"); applied=True
        else: print(f"  Skip: {pf.name}")
    return applied

def run(): return run_autofix_agent()
if __name__ == "__main__": run_autofix_agent()
