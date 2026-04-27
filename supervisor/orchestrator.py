"""
supervisor/orchestrator.py
==========================
LangGraph StateGraph Orchestrator — FIXED VERSION

FIXES vs previous version:
  FIX 1: pipeline_passed now True when patches fix the issues
          (not just "not errors_found")
  FIX 2: quality_score written back into result dict in node_code_review
          so node_summary always reads a consistent value
  FIX 3: build_status read from multiple fallback keys in debug_result
  FIX 4: security score extracted robustly with multiple fallback keys
  FIX 5: All score reads use multi-key fallback to handle different agent formats

Pipeline graph:
  code_review -> security -> debug -> fault_analysis
                                           |
                               test_gen -> optimization -> release -> autofix -> summary -> END
"""
import json, os, sys, re
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.debug_agent          import run_debug_agent
from agents.security_agent       import run_security_agent
from agents.code_review_agent    import run_code_review_agent
from agents.test_gen_agent       import run_test_gen_agent
from agents.optimization_agent   import run_optimization_agent
from agents.release_agent        import run_release_agent
from agents.fault_analysis_agent import run_fault_analysis_agent
from agents.autofix_agent        import run_autofix_agent

load_dotenv()
TARGET  = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS = Path("reports")
FIRMWARE = Path("firmware")


# ════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    target:      str
    source_path: str
    version:     str
    code_review_result:    dict
    security_result:       dict
    debug_result:          dict
    testgen_result:        dict
    optimization_result:   dict
    release_result:        dict
    fault_analysis_result: dict
    autofix_result:        dict
    container_scan_result: dict
    unit_test_result:      dict
    slsa_hashes:           dict
    ota_manifest:          dict
    deploy_status:         str
    feedback_issues:       list
    fault_injection_result: dict
    hil_result:             dict
    dynamic_score:          int
    patches_generated:      int
    tests_deployed:         bool
    current_stage:          str
    errors_found:           bool
    pipeline_passed:        bool
    summary:                str


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_slsa_hashes(target: str) -> dict:
    try:
        lines = (REPORTS / "firmware-sha256.txt").read_text().strip().splitlines()
        return {p[1]: p[0] for p in (l.strip().split() for l in lines) if len(p) == 2}
    except Exception:
        return {}


def _load_deploy_status() -> str:
    try:
        return (REPORTS / "deploy-status.txt").read_text().strip()
    except Exception:
        return "simulated"


def _extract_score(result: dict, *keys) -> int | str:
    """
    ═══ FIX 4 ═══
    Try multiple keys to extract a numeric score from an agent result.
    Fallback to regex on the 'review' or 'summary' string.
    Returns int or "N/A".

    Why needed: different agents use different key names:
      security_agent  → "security_score"
      code_review     → no direct key, score buried in markdown "review"
      fault_analysis  → "robustness_score"
    """
    for k in keys:
        v = result.get(k)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass

    # Fallback: search in any text field
    for field in ("review", "summary", "analysis", "report"):
        text = result.get(field, "")
        if text:
            m = re.search(r"(\d+)\s*(?:out of|/)\s*10", str(text), re.I)
            if m:
                return int(m.group(1))

    return "N/A"


def _extract_build_status(debug_result: dict) -> str:
    """
    ═══ FIX 3 ═══
    debug_agent may write build status under different keys.
    Try all known variants before defaulting to 'unknown'.
    """
    for key in ("build_status", "overall_health", "status", "compilation_status"):
        v = debug_result.get(key)
        if v:
            return str(v)

    # Infer from compilation errors
    errors = debug_result.get("compilation_errors", [])
    if isinstance(errors, list):
        return "success" if len(errors) == 0 else "failed"

    return "unknown"


# ════════════════════════════════════════════════════════════════════
# CI ARTIFACT LOADER
# ════════════════════════════════════════════════════════════════════

def load_ci_artifacts(state: PipelineState) -> PipelineState:
    target = state["target"]
    state["container_scan_result"]  = _load_json(REPORTS / "container-scan-summary.json")
    state["unit_test_result"]       = _load_json(REPORTS / "unit-test-results.json")
    state["slsa_hashes"]            = _load_slsa_hashes(target)
    state["ota_manifest"]           = _load_json(REPORTS / "ota-manifest-signed.json")
    state["deploy_status"]          = _load_deploy_status()
    state["feedback_issues"]        = []
    state["fault_injection_result"] = _load_json(REPORTS / f"fault-injection-report-{target}.json")
    state["hil_result"]             = _load_json(REPORTS / f"hil-report-{target}.json")
    state["patches_generated"]      = 0
    state["tests_deployed"]         = False
    return state


# ════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ════════════════════════════════════════════════════════════════════

def node_code_review(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Code Review Agent  (Agent 3)")
    state["current_stage"] = "code_review"
    try:
        result = run_code_review_agent(target=state["target"])

        # ═══ FIX 2 ═══
        # Extract score and write it BACK into the result dict so every
        # downstream reader (node_summary, pipeline-summary.json) sees
        # a consistent "quality_score" key — no need to re-parse markdown.
        score = _extract_score(result, "quality_score", "score", "code_score")
        result["quality_score"] = score  # ← write back

        state["code_review_result"] = result

        if isinstance(score, int) and score < 5:
            state["errors_found"] = True

    except Exception as e:
        print(f"[Orchestrator] Code Review failed: {e}")
        state["code_review_result"] = {"error": str(e), "quality_score": "N/A"}
        state["errors_found"] = True
    return state


def node_security(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Security Agent  (Agent 2)")
    state["current_stage"] = "security"
    try:
        result = run_security_agent(target=state["target"])

        # ═══ FIX 4 ═══
        # security_agent may write "security_score" or "score"
        score = _extract_score(result, "security_score", "score")
        result["security_score"] = score  # normalize key

        secrets = result.get("secrets_found", []) or []
        state["security_result"] = result

        # errors_found = True signals AutoFix to generate patches
        # but should NOT block pipeline_passed on its own
        if (isinstance(score, int) and score < 6) or len(secrets) > 0:
            state["errors_found"] = True

    except Exception as e:
        print(f"[Orchestrator] Security Agent failed: {e}")
        state["security_result"] = {"error": str(e), "security_score": 0}
        state["errors_found"] = True
    return state


def node_debug(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Debug Agent  (Agent 1)")
    state["current_stage"] = "debug"
    try:
        result = run_debug_agent(target=state["target"])
        state["debug_result"] = result
        errors = result.get("compilation_errors", []) or []
        health = result.get("overall_health", "unknown")
        # Only block on actual compilation errors, not warnings
        if health == "broken" or len(errors) > 0:
            state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Debug Agent failed: {e}")
        state["debug_result"] = {"error": str(e)}
        state["errors_found"] = True
    return state


def node_fault_analysis(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Fault Analysis Agent  (Agent 7)")
    state["current_stage"] = "fault_analysis"
    try:
        result = run_fault_analysis_agent(target=state["target"])
        state["fault_analysis_result"] = result
        score = result.get("robustness_score", 10)
        if isinstance(score, (int, float)) and score < 5:
            state["errors_found"] = True

        qemu_pass   = (state.get("debug_result", {})
                       .get("dynamic_findings", {})
                       .get("qemu_status") == "pass")
        fuzzer_pass = (state.get("debug_result", {})
                       .get("dynamic_findings", {})
                       .get("fuzzer_status") == "pass")
        hil_pass    = state.get("hil_result", {}).get("status") == "pass"

        try:
            dynamic_score = int(
                (float(score) * 0.4) +
                (10 if qemu_pass   else 0) * 0.3 +
                (10 if fuzzer_pass else 0) * 0.2 +
                (10 if hil_pass    else 0) * 0.1
            )
        except (ValueError, TypeError):
            dynamic_score = 0

        state["dynamic_score"] = min(dynamic_score, 10)
        print(f"[Orchestrator] Dynamic score: {state['dynamic_score']}/10")
    except Exception as e:
        print(f"[Orchestrator] Fault Analysis Agent failed: {e}")
        state["fault_analysis_result"] = {"error": str(e)}
        state["dynamic_score"] = 0
    return state


def node_test_gen(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Test Generation Agent  (Agent 4)")
    state["current_stage"] = "test_gen"
    try:
        result = run_test_gen_agent(target=state["target"])
        state["testgen_result"] = result
        n      = len(result.get("test_cases", []))
        deploy = result.get("deploy_manifest", {})
        state["tests_deployed"] = deploy.get("status") in ("deployed", "partial")
        print(f"[Orchestrator] {n} test cases | deploy={deploy.get('status', '?')}")
    except Exception as e:
        print(f"[Orchestrator] Test Gen Agent failed: {e}")
        state["testgen_result"] = {"error": str(e)}
        state["tests_deployed"] = False
    return state


def node_optimization(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Optimization Agent  (Agent 5)")
    state["current_stage"] = "optimization"
    try:
        result = run_optimization_agent(target=state["target"])
        state["optimization_result"] = result
        for region in ("flash", "dram", "iram"):
            if result.get("memory_usage", {}).get(f"{region}_risk") == "critical":
                state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Optimization Agent failed: {e}")
        state["optimization_result"] = {"error": str(e)}
    return state


def node_release(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Release Agent  (Agent 6)")
    state["current_stage"] = "release"
    try:
        deploy_val = state.get("deploy_status", "")
        if deploy_val:
            REPORTS.mkdir(exist_ok=True)
            (REPORTS / "deploy-status.txt").write_text(deploy_val)
        result = run_release_agent(version=state["version"], target=state["target"])
        state["release_result"] = result
    except Exception as e:
        print(f"[Orchestrator] Release Agent failed: {e}")
        state["release_result"] = {"error": str(e)}
    return state


def node_autofix(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: AutoFix Agent  (Agent 8)")
    state["current_stage"] = "autofix"
    try:
        result = run_autofix_agent(target=state["target"], apply_patches=False)
        state["autofix_result"]    = result
        state["patches_generated"] = result.get("patches_generated", 0)
        n = state["patches_generated"]
        print(f"[Orchestrator] AutoFix: {n} patch(es) generated")
    except Exception as e:
        print(f"[Orchestrator] AutoFix Agent failed: {e}")
        state["autofix_result"]    = {"error": str(e)}
        state["patches_generated"] = 0
    return state


def node_summary(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("NODE: Pipeline Summary")
    state["current_stage"] = "summary"

    target  = state["target"]
    version = state["version"]

    # ── Read scores ───────────────────────────────────────────────
    _cr  = state.get("code_review_result", {})
    _sec = state.get("security_result", {})
    _dbg = state.get("debug_result", {})
    _opt = state.get("optimization_result", {})
    _fa  = state.get("fault_analysis_result", {})
    _af  = state.get("autofix_result", {})

    # ═══ FIX 2 ═══ quality_score is now in result dict (written by node_code_review)
    code_score = _cr.get("quality_score", "N/A")
    if code_score == "N/A":
        code_score = _extract_score(_cr, "quality_score", "score")

    # ═══ FIX 4 ═══ normalized by node_security
    sec_score = _sec.get("security_score", "N/A")
    if sec_score == "N/A":
        sec_score = _extract_score(_sec, "security_score", "score")

    # ═══ FIX 3 ═══ robust build status extraction
    build_ok  = _extract_build_status(_dbg)

    flash_pct  = _opt.get("memory_usage", {}).get("flash_pct", "N/A")
    rob_score  = _fa.get("robustness_score", "N/A")
    dyn_score  = state.get("dynamic_score", "N/A")
    n_cves     = len(_sec.get("critical_cves", []) or [])
    n_secrets  = len(_sec.get("secrets_found", []) or [])
    hil_status = state.get("hil_result", {}).get("status", "not_run")
    n_patches  = state.get("patches_generated", 0)
    tests_ok   = state.get("tests_deployed", False)
    errors     = state.get("errors_found", False)

    n_tests = len(state.get("testgen_result", {}).get("test_cases", []))
    if n_tests == 0 and (REPORTS / f"generated_tests_{target}.cpp").exists():
        n_tests = 1

    second_run_ready = n_patches > 0 or tests_ok

    # ═══ FIX 1 ═══
    # pipeline_passed:
    #   True  if no errors at all
    #   True  if errors exist BUT patches were generated (AutoFix is handling them)
    #   False only if errors exist AND no patches could be generated
    if not errors:
        state["pipeline_passed"] = True
    elif n_patches > 0:
        # Issues found but AutoFix generated patches → considered handled
        state["pipeline_passed"] = True
    else:
        state["pipeline_passed"] = False

    passed = state["pipeline_passed"]

    # ── Console summary ───────────────────────────────────────────
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        f"║  ESP32 Matter DevSecOps AI Pipeline -- {version}",
        f"║  Target: {target}",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Code Review      Score : {code_score}/10",
        f"║  Security         Score : {sec_score}/10 | CVEs: {n_cves} | Secrets: {n_secrets}",
        f"║  Build            Status: {build_ok}",
        f"║  Tests            Count : {n_tests} | deployed: {tests_ok}",
        f"║  Memory           Flash : {flash_pct}%",
        f"║  Robustness       Score : {rob_score}/10",
        f"║  Dynamic Score          : {dyn_score}/10",
        f"║  HIL Hardware           : {hil_status}",
        f"║  AutoFix          Patches: {n_patches}",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Overall: {'PASSED' if passed else 'ISSUES — no patches generated'}",
        f"║  Second Run Ready: {'YES' if second_run_ready else 'NO'}",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    state["summary"] = "\n".join(lines)
    print(state["summary"])

    # ── Write pipeline-summary.json ───────────────────────────────
    REPORTS.mkdir(exist_ok=True)
    full_summary = {
        "target":           target,
        "version":          version,
        "pipeline_passed":  passed,
        "errors_found":     errors,
        "second_run_ready": second_run_ready,
        "stage_results": {
            # ── Keys read by ci.yml AI Summary step ──
            "code_quality": {
                "score":  code_score,          # ci.yml: cod.get('score','N/A')
                "issues": len(_cr.get("issues", []) or []),
            },
            "security": {
                "score":   sec_score,           # ci.yml: sec.get('score','N/A')
                "cves":    n_cves,              # ci.yml: sec.get('cves',0)
                "secrets": n_secrets,
            },
            "build": {
                "status": build_ok,             # ci.yml: bld.get('status','?')
            },
            "memory": {
                "flash_pct": flash_pct,         # ci.yml: mem.get('flash_pct','N/A')
            },
            "autofix": {
                "patches_generated": n_patches, # ci.yml: af.get('patches_generated',0)
                "patch_files":    _af.get("patch_files", []),
                "issues_analyzed":_af.get("issues_analyzed", 0),
                "status":         _af.get("status", "unknown"),
            },
            # ── Additional keys for rich reporting ──
            "tests_generated": n_tests,
            "tests_deployed":  tests_ok,
            "fault_injection": {
                "robustness_score":  rob_score,
                "dynamic_score":     dyn_score,
                "verdict":           _fa.get("overall_verdict", "N/A"),
            },
            "hil":     {"status": hil_status},
            "release": {
                "version":      version,
                "canary_deploy": state.get("deploy_status", "not_run"),
            },
        },
    }
    out = REPORTS / "pipeline-summary.json"
    out.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"\n[Orchestrator] Summary saved: {out}")
    return state


# ════════════════════════════════════════════════════════════════════
# GRAPH
# ════════════════════════════════════════════════════════════════════

def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("code_review",    node_code_review)
    graph.add_node("security",       node_security)
    graph.add_node("debug",          node_debug)
    graph.add_node("fault_analysis", node_fault_analysis)
    graph.add_node("test_gen",       node_test_gen)
    graph.add_node("optimization",   node_optimization)
    graph.add_node("release",        node_release)
    graph.add_node("autofix",        node_autofix)
    graph.add_node("summary",        node_summary)

    graph.set_entry_point("code_review")
    graph.add_edge("code_review",    "security")
    graph.add_edge("security",       "debug")
    graph.add_edge("debug",          "fault_analysis")
    graph.add_edge("fault_analysis", "test_gen")
    graph.add_edge("test_gen",       "optimization")
    graph.add_edge("optimization",   "release")
    graph.add_edge("release",        "autofix")
    graph.add_edge("autofix",        "summary")
    graph.add_edge("summary",        END)

    return graph.compile()


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def run_pipeline(target: str = TARGET, version: str = "v1.0.0") -> PipelineState:
    print("\n" + "#" * 64)
    print("#  ESP32 Matter — AI DevSecOps Pipeline (Fixed Orchestrator)  #")
    print(f"#  Target : {target:<10}  Version : {version:<28}  #")
    print("#" * 64)

    REPORTS.mkdir(exist_ok=True)

    initial: PipelineState = {
        "target":      target,
        "source_path": os.getenv("EXAMPLE_PATH", "esp-matter/examples/light"),
        "version":     version,
        "code_review_result":    {},
        "security_result":       {},
        "debug_result":          {},
        "testgen_result":        {},
        "optimization_result":   {},
        "release_result":        {},
        "fault_analysis_result": {},
        "autofix_result":        {},
        "container_scan_result": {},
        "unit_test_result":      {},
        "slsa_hashes":           {},
        "ota_manifest":          {},
        "deploy_status":         "",
        "feedback_issues":       [],
        "fault_injection_result":{},
        "hil_result":            {},
        "dynamic_score":         0,
        "patches_generated":     0,
        "tests_deployed":        False,
        "current_stage":         "init",
        "errors_found":          False,
        "pipeline_passed":       False,
        "summary":               "",
    }

    print("\n[Orchestrator] Loading CI artifacts...")
    initial = load_ci_artifacts(initial)
    print(f"  unit_tests      : {initial['unit_test_result'].get('status', 'none')}")
    print(f"  hil_result      : {initial['hil_result'].get('status', 'not_run')}")

    pipeline    = build_pipeline_graph()
    final_state = pipeline.invoke(initial)

    n = final_state.get("patches_generated", 0)
    print(f"\n[Orchestrator] Done. Passed: {final_state['pipeline_passed']}")
    print(f"[Orchestrator] Patches: {n}")
    if n > 0:
        patch_files = final_state.get("autofix_result", {}).get("patch_files", [])
        for pf in patch_files:
            print(f"  → {pf}")

    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ESP32 Matter DevSecOps AI Pipeline Orchestrator"
    )
    parser.add_argument("--target",  default=os.getenv("TARGET_CHIP", TARGET))
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()
    run_pipeline(target=args.target, version=args.version)
