"""
supervisor/orchestrator.py
==========================
LangGraph StateGraph Orchestrator — version finale avec Agent 8 (AutoFix).

Pipeline graph :
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
TARGET   = os.getenv("TARGET_CHIP", "esp32c3")
REPORTS  = Path("reports")
FIRMWARE = Path("firmware")


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
    patches_generated:  int
    tests_deployed:     bool
    current_stage:   str
    errors_found:    bool
    pipeline_passed: bool
    summary:         str


def _load_json(path: Path) -> dict:
    try:    return json.loads(path.read_text(encoding="utf-8"))
    except: return {}

def _load_slsa_hashes(target: str) -> dict:
    try:
        lines = (REPORTS / "firmware-sha256.txt").read_text().strip().splitlines()
        return {p[1]: p[0] for p in (l.strip().split() for l in lines) if len(p) == 2}
    except: return {}

def _load_deploy_status() -> str:
    try:    return (REPORTS / "deploy-status.txt").read_text().strip()
    except: return "simulated"


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


def node_code_review(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Code Review Agent  (Agent 3)")
    state["current_stage"] = "code_review"
    try:
        result = run_code_review_agent(target=state["target"])
        state["code_review_result"] = result
        score = result.get("quality_score", None)
        if score is None:
            m = re.search(r"(\d+)\s*(?:out of|/)\s*10", result.get("review", ""), re.I)
            score = int(m.group(1)) if m else None
        if score is not None and score < 5:
            state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Code Review failed: {e}")
        state["code_review_result"] = {"error": str(e)}
        state["errors_found"] = True
    return state


def node_security(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Security Agent  (Agent 2)")
    state["current_stage"] = "security"
    try:
        result = run_security_agent(target=state["target"])
        state["security_result"] = result
        score   = result.get("security_score", 10)
        secrets = len(result.get("secrets_found", []))
        if score < 6 or secrets > 0:
            state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Security Agent failed: {e}")
        state["security_result"] = {"error": str(e), "security_score": 0}
        state["errors_found"] = True
    return state


def node_debug(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Debug Agent  (Agent 1)")
    state["current_stage"] = "debug"
    try:
        result = run_debug_agent(target=state["target"])
        state["debug_result"] = result
        errors = len(result.get("compilation_errors", []))
        health = result.get("overall_health", "unknown")
        if health == "broken" or errors > 0:
            state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Debug Agent failed: {e}")
        state["debug_result"] = {"error": str(e)}
        state["errors_found"] = True
    return state


def node_fault_analysis(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Fault Analysis Agent  (Agent 7)")
    state["current_stage"] = "fault_analysis"
    try:
        result = run_fault_analysis_agent(target=state["target"])
        state["fault_analysis_result"] = result
        score = result.get("robustness_score", 10)
        if score < 5:
            state["errors_found"] = True

        qemu_pass   = state.get("debug_result", {}).get("dynamic_findings", {}).get("qemu_status") == "pass"
        fuzzer_pass = state.get("debug_result", {}).get("dynamic_findings", {}).get("fuzzer_status") == "pass"
        fault_pass  = state.get("fault_injection_result", {}).get("overall_status") == "pass"
        hil_pass    = state.get("hil_result", {}).get("status") == "pass"

        dynamic_score = int(
            (score * 0.4) +
            (10 if qemu_pass   else 0) * 0.3 +
            (10 if fuzzer_pass else 0) * 0.2 +
            (10 if hil_pass    else 0) * 0.1
        )
        state["dynamic_score"] = min(dynamic_score, 10)
        print(f"[Orchestrator] Dynamic score: {state['dynamic_score']}/10")
    except Exception as e:
        print(f"[Orchestrator] Fault Analysis Agent failed: {e}")
        state["fault_analysis_result"] = {"error": str(e)}
        state["dynamic_score"]         = 0
    return state


def node_test_gen(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Test Generation Agent  (Agent 4)")
    state["current_stage"] = "test_gen"
    try:
        result = run_test_gen_agent(target=state["target"])
        state["testgen_result"] = result
        n      = len(result.get("test_cases", []))
        deploy = result.get("deploy_manifest", {})
        state["tests_deployed"] = deploy.get("status") in ("deployed", "partial")
        print(f"[Orchestrator] {n} test cases | deploy={deploy.get('status','?')}")
    except Exception as e:
        print(f"[Orchestrator] Test Gen Agent failed: {e}")
        state["testgen_result"] = {"error": str(e)}
        state["tests_deployed"] = False
    return state


def node_optimization(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Optimization Agent  (Agent 5)")
    state["current_stage"] = "optimization"
    try:
        result = run_optimization_agent(target=state["target"])
        state["optimization_result"] = result
        for region in ["flash", "dram", "iram"]:
            if result.get("memory_usage", {}).get(f"{region}_risk") == "critical":
                state["errors_found"] = True
    except Exception as e:
        print(f"[Orchestrator] Optimization Agent failed: {e}")
        state["optimization_result"] = {"error": str(e)}
    return state


def node_release(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
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
    print("\n" + "="*60)
    print("NODE: AutoFix Agent  (Agent 8 — patches + test deploy)")
    state["current_stage"] = "autofix"
    try:
        result = run_autofix_agent(target=state["target"], apply_patches=False)
        state["autofix_result"]    = result
        state["patches_generated"] = result.get("patches_generated", 0)

        test_int = result.get("test_integration", {})
        if test_int.get("status") == "deployed" and not state.get("tests_deployed"):
            state["tests_deployed"] = True

        n = state["patches_generated"]
        print(f"[Orchestrator] AutoFix: {n} patch(es) generated")
    except Exception as e:
        print(f"[Orchestrator] AutoFix Agent failed: {e}")
        state["autofix_result"]    = {"error": str(e)}
        state["patches_generated"] = 0
    return state


def node_summary(state: PipelineState) -> PipelineState:
    print("\n" + "="*60)
    print("NODE: Pipeline Summary")
    state["current_stage"] = "summary"

    target  = state["target"]
    version = state["version"]

    _cr        = state.get("code_review_result", {})
    code_score = _cr.get("quality_score", None)
    if code_score is None:
        m = re.search(r"(\d+)\s*(?:out of|/)\s*10", _cr.get("review", ""), re.I)
        code_score = int(m.group(1)) if m else "N/A"

    sec_score  = state.get("security_result", {}).get("security_score", "N/A")
    build_ok   = state.get("debug_result", {}).get("build_status", "unknown")
    flash_pct  = state.get("optimization_result", {}).get("memory_usage", {}).get("flash_pct", "N/A")
    rob_score  = state.get("fault_analysis_result", {}).get("robustness_score", "N/A")
    dyn_score  = state.get("dynamic_score", "N/A")
    n_cves     = len(state.get("security_result", {}).get("critical_cves", []))
    hil_status = state.get("hil_result", {}).get("status", "not_run")
    n_patches  = state.get("patches_generated", 0)
    tests_ok   = state.get("tests_deployed", False)
    errors     = state.get("errors_found", False)

    _cpp_file  = REPORTS / f"generated_tests_{target}.cpp"
    n_tests    = len(state.get("testgen_result", {}).get("test_cases", []))
    n_tests    = n_tests if n_tests > 0 else (1 if _cpp_file.exists() else 0)

    second_run_ready   = n_patches > 0 or tests_ok
    state["pipeline_passed"] = not errors

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        f"║  ESP32 Matter DevSecOps AI Pipeline -- {version}",
        f"║  Target: {target}",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Agent 3 -- Code Quality      Score: {code_score}/10",
        f"║  Agent 2 -- Security          Score: {sec_score}/10 | CVEs: {n_cves}",
        f"║  Agent 1 -- Build             Status: {build_ok}",
        f"║  Agent 4 -- Tests             {n_tests} cases | deployed: {tests_ok}",
        f"║  Agent 5 -- Memory            Flash: {flash_pct}%",
        f"║  Agent 7 -- Robustness        Score: {rob_score}/10",
        f"║  Dynamic Score (composite)    {dyn_score}/10",
        f"║  HIL Real Hardware            {hil_status}",
        f"║  Agent 8 -- AutoFix           {n_patches} patch(es)",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Overall: {'PASSED' if not errors else 'ISSUES DETECTED'}",
        f"║  Second Run Ready: {'YES' if second_run_ready else 'NO'}",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    state["summary"] = "\n".join(lines)
    print(state["summary"])

    REPORTS.mkdir(exist_ok=True)
    full_summary = {
        "target":           target,
        "version":          version,
        "pipeline_passed":  state["pipeline_passed"],
        "errors_found":     errors,
        "second_run_ready": second_run_ready,
        "stage_results": {
            "code_quality":    {"score": code_score, "issues": len(_cr.get("issues", []))},
            "security":        {"score": sec_score, "cves": n_cves,
                                "secrets": len(state.get("security_result", {}).get("secrets_found", []))},
            "build":           {"status": build_ok},
            "tests_generated": n_tests,
            "tests_deployed":  tests_ok,
            "memory":          {"flash_pct": flash_pct},
            "fault_injection": {
                "robustness_score":  rob_score,
                "verdict":           state.get("fault_analysis_result", {}).get("overall_verdict", "N/A"),
                "critical_failures": state.get("fault_injection_result", {}).get("critical_failures", []),
            },
            "dynamic_score": dyn_score,
            "hil":           {"status": hil_status,
                              "boot_success": state.get("hil_result", {}).get("boot_success", False),
                              "matter_started": state.get("hil_result", {}).get("matter_started", False)},
            "autofix": {
                "patches_generated":  n_patches,
                "patch_files":        state.get("autofix_result", {}).get("patch_files", []),
                "apply_script":       state.get("autofix_result", {}).get("apply_script", ""),
                "test_deploy_status": state.get("autofix_result", {}).get("test_integration", {}).get("status", "N/A"),
                "issues_analyzed":    state.get("autofix_result", {}).get("issues_analyzed", 0),
            },
            "release":       {"version": version, "canary_deploy": state.get("deploy_status", "not_run")},
        },
    }
    (REPORTS / "pipeline-summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"\n[Orchestrator] Summary saved: {REPORTS / 'pipeline-summary.json'}")
    return state


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


def run_pipeline(target: str = TARGET, version: str = "v1.0.0") -> PipelineState:
    print("\n" + "#"*64)
    print("#  ESP32 Matter -- AI DevSecOps Pipeline v3 (with AutoFix) #")
    print(f"#  Target: {target:<10}  Version: {version:<28}  #")
    print("#"*64)
    REPORTS.mkdir(exist_ok=True)
    initial: PipelineState = {
        "target": target, "source_path": os.getenv("EXAMPLE_PATH", "esp-matter/examples/light"),
        "version": version, "code_review_result": {}, "security_result": {},
        "debug_result": {}, "testgen_result": {}, "optimization_result": {},
        "release_result": {}, "fault_analysis_result": {}, "autofix_result": {},
        "container_scan_result": {}, "unit_test_result": {}, "slsa_hashes": {},
        "ota_manifest": {}, "deploy_status": "", "feedback_issues": [],
        "fault_injection_result": {}, "hil_result": {}, "dynamic_score": 0,
        "patches_generated": 0, "tests_deployed": False,
        "current_stage": "init", "errors_found": False,
        "pipeline_passed": False, "summary": "",
    }
    print("\n[Orchestrator] Loading CI artifacts...")
    initial = load_ci_artifacts(initial)
    print(f"  fault_injection : {bool(initial['fault_injection_result'])}")
    print(f"  hil_result      : {initial['hil_result'].get('status', 'not_run')}")
    pipeline    = build_pipeline_graph()
    final_state = pipeline.invoke(initial)
    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  default=TARGET)
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()
    final = run_pipeline(target=args.target, version=args.version)
    n = final.get("patches_generated", 0)
    print(f"\nPipeline complete. Passed: {final['pipeline_passed']}")
    print(f"Patches generated: {n}")
    if n > 0:
        print("Apply: bash reports/patches/APPLY_ALL.sh")
