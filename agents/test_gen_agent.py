"""
agents/test_gen_agent.py  —  Agent 4 : Test Generation Agent
=============================================================
ROLE:
  Generates Unity-based C++ unit tests for the ESP-Matter light
  firmware and deploys them into the ESP-Matter test directory so
  the SECOND CI run compiles and runs them automatically.

KEY BEHAVIOURS:
  1. Robust JSON parsing (3-level fallback) — handles C++ code inside
     JSON with unescaped newlines (original design kept intact).
  2. Saves generated_tests_{target}.cpp to reports/ for artifact upload.
  3. DEPLOYS the test file into esp-matter/examples/light/test/ so
     the next build finds it without manual steps.
  4. Saves a test-CMakeLists.txt next to the test file so idf.py
     can compile it automatically.
  5. Writes a deploy-manifest.json that the CI "Apply AutoFix patches"
     step can read to confirm test deployment status.

INPUTS:
  reports/unit-test-results.json     (from Stage 5 CI)
  esp-matter/examples/light/main/    (source files)
  reports/fault-analysis-report-{target}.json  (for robustness tests)
  reports/code-review-{target}.json            (for regression targets)

OUTPUTS:
  reports/testgen-report-{target}.json
  reports/generated_tests_{target}.cpp
  esp-matter/examples/light/test/test_light_app_{target}.cpp  ← NEW
  esp-matter/examples/light/test/CMakeLists.txt               ← NEW
  reports/test-deploy-manifest.json                           ← NEW
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TARGET      = os.getenv("TARGET_CHIP", "esp32c3")
SOURCE_BASE = Path(os.getenv("EXAMPLE_PATH",
                             "esp-matter/examples/light/main"))
REPORTS     = Path("reports")

SOURCE_FILES = ["app_main.cpp", "app_driver.cpp", "app_priv.h"]

CANDIDATE_DIRS = [
    SOURCE_BASE / "main",
    Path.home() / "esp-matter" / "examples" / "light" / "main",
    Path("/opt/espressif/esp-matter/examples/light/main"),
    Path("esp-matter/examples/light/main"),
    Path("../esp-matter/examples/light/main"),
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_source_dir() -> Path | None:
    for c in CANDIDATE_DIRS:
        r = c.resolve()
        if r.is_dir() and (list(r.glob("*.cpp")) + list(r.glob("*.c"))):
            return r
    return None


def _load_source_files() -> dict[str, str]:
    src_dir = _find_source_dir()
    sources = {}
    if src_dir:
        for name in SOURCE_FILES:
            p = src_dir / name
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if content:
                    sources[name] = content
            except Exception:
                pass
    return sources


def _summarise_existing_tests(result: dict) -> str:
    if not result:
        return "No existing test data."
    if result.get("status") == "no_tests_yet":
        return (
            "No tests have been implemented yet.\n"
            f"Note: {result.get('note', '')}\n"
            "Generate a COMPLETE test suite from scratch."
        )
    return (
        f"Existing: passed={result.get('passed', 0)}, "
        f"failed={result.get('failed', 0)}\n"
        "Generate ADDITIONAL tests not already covered."
    )


def _summarise_fault_context(fault: dict) -> str:
    """Extract robustness scenarios to generate targeted regression tests."""
    if not fault:
        return "No fault analysis data available."
    failed = fault.get("failed_scenarios_analysis", [])
    passed = fault.get("passed_scenarios_highlights", [])
    lines = [f"Robustness score: {fault.get('robustness_score', 'N/A')}/10"]
    if failed:
        lines.append("Failed scenarios needing regression tests:")
        for s in failed[:4]:
            lines.append(f"  - {s.get('scenario', '?')}: {s.get('root_cause', '')[:120]}")
    if passed:
        lines.append("Passing scenarios to keep in regression suite:")
        for s in passed[:3]:
            lines.append(f"  - {s.get('scenario', '?')}: {s.get('why_good', '')[:80]}")
    return "\n".join(lines)


def _summarise_code_issues(review: dict) -> str:
    """Highlight code review issues to generate targeted tests for them."""
    issues = review.get("issues", [])
    if not issues:
        return "No code review issues available."
    high = [i for i in issues if i.get("severity") in ("critical", "high")][:4]
    if not high:
        return "No high-severity issues from code review."
    lines = ["High-severity issues to cover with tests:"]
    for i in high:
        lines.append(f"  - [{i.get('severity')}] {i.get('description', '')[:120]}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# ROBUST JSON PARSING  (3-level fallback — original logic kept)
# ═══════════════════════════════════════════════════════════════

def _robust_parse(raw: str, target: str) -> dict:
    """
    Three-level fallback parser for LLM JSON responses that contain
    C++ code blocks with unescaped newlines.
    """
    # Level 1 — standard parse after stripping markdown fences
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    clean = clean.rstrip("`").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Level 2 — escape literal newlines inside JSON string values
    try:
        fixed = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace('\n', '\\n').replace('\r', '\\r'),
            clean,
            flags=re.DOTALL,
        )
        return json.loads(fixed)
    except Exception:
        pass

    # Level 3 — manual field extraction
    print("[Test Gen Agent] Attempting manual field extraction...")
    report = {
        "target":          target,
        "parse_error":     True,
        "parse_recovered": True,
    }

    cpp_match = re.search(
        r'"test_file_content"\s*:\s*"(.*?)(?<!\\)"(?=\s*[,}])',
        clean, re.DOTALL,
    )
    if cpp_match:
        cpp = cpp_match.group(1)
        cpp = cpp.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        report["test_file_content"] = cpp

    sum_match = re.search(r'"summary"\s*:\s*"(.*?)(?<!\\)"', clean, re.DOTALL)
    if sum_match:
        report["summary"] = sum_match.group(1).replace('\\n', ' ')

    cases = re.findall(
        r'"name"\s*:\s*"([^"]+)".*?"type"\s*:\s*"([^"]+)".*?"area"\s*:\s*"([^"]+)"',
        clean, re.DOTALL,
    )
    if cases:
        report["test_cases"] = [
            {"name": n, "type": t, "area": a} for n, t, a in cases
        ]

    mocks = re.findall(
        r'"filename"\s*:\s*"([^"]+)".*?"content"\s*:\s*"(.*?)(?<!\\)"(?=\s*\})',
        clean, re.DOTALL,
    )
    if mocks:
        report["mock_files"] = [
            {
                "filename": fn,
                "content":  content.replace('\\n', '\n').replace('\\"', '"'),
            }
            for fn, content in mocks
        ]

    if "test_file_content" not in report:
        report["raw_response"] = raw

    return report


# ═══════════════════════════════════════════════════════════════
# TEST DEPLOYMENT  (NEW — deploys tests for second CI run)
# ═══════════════════════════════════════════════════════════════

_CMAKE_TEMPLATE = """\
# CMakeLists.txt — auto-generated by test_gen_agent
# Enables Unity tests to be compiled via idf.py build --target {target}
cmake_minimum_required(VERSION 3.16)
include($ENV{{IDF_PATH}}/tools/cmake/project.cmake)
project(light_test)
"""

_CMAKELISTS_COMPONENT = """\
# idf_component_register for the test component
idf_component_register(
    SRCS "test_light_app_{target}.cpp"
    INCLUDE_DIRS "."
    REQUIRES unity esp_system
)
"""

def _deploy_test_file(
    cpp_content: str,
    target: str,
    mock_files: list[dict],
) -> dict:
    """
    Deploy the generated test file and mocks into the ESP-Matter
    test directory.  Also writes CMakeLists.txt so idf.py finds it.

    Returns a manifest dict describing what was deployed.
    """
    src_dir = _find_source_dir()
    if src_dir is None:
        return {
            "status": "source_not_found",
            "note":   "ESP-Matter source not found; test file kept in reports/ only",
        }

    # reports/ already has the .cpp — now also put it in the test dir
    test_dir = src_dir.parent / "test"
    test_dir.mkdir(exist_ok=True)

    deployed_files = []
    errors = []

    # Main test file
    test_cpp_path = test_dir / f"test_light_app_{target}.cpp"
    try:
        test_cpp_path.write_text(cpp_content, encoding="utf-8")
        deployed_files.append(str(test_cpp_path))
    except Exception as e:
        errors.append(f"test .cpp: {e}")

    # CMakeLists.txt for the project level
    cmake_proj = test_dir / "CMakeLists.txt"
    try:
        cmake_proj.write_text(
            _CMAKE_TEMPLATE.format(target=target), encoding="utf-8"
        )
        deployed_files.append(str(cmake_proj))
    except Exception as e:
        errors.append(f"CMakeLists.txt: {e}")

    # main/ component CMakeLists
    main_comp_dir = test_dir / "main"
    main_comp_dir.mkdir(exist_ok=True)
    comp_cmake = main_comp_dir / "CMakeLists.txt"
    try:
        comp_cmake.write_text(
            _CMAKELISTS_COMPONENT.format(target=target), encoding="utf-8"
        )
        deployed_files.append(str(comp_cmake))
    except Exception as e:
        errors.append(f"main/CMakeLists.txt: {e}")

    # Mock files
    for mock in mock_files:
        fn   = mock.get("filename", "")
        cont = mock.get("content", "")
        if fn and cont:
            mock_path = test_dir / fn
            try:
                mock_path.write_text(cont, encoding="utf-8")
                deployed_files.append(str(mock_path))
            except Exception as e:
                errors.append(f"{fn}: {e}")

    manifest = {
        "status":         "deployed" if not errors else "partial",
        "test_dir":       str(test_dir),
        "deployed_files": deployed_files,
        "errors":         errors,
        "timestamp":      datetime.now().isoformat(),
    }

    # Write deploy manifest for CI to read
    manifest_path = REPORTS / "test-deploy-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ═══════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_test_gen_agent(target: str = TARGET) -> dict:
    print(f"\n[Test Gen Agent] Starting for target: {target}")

    unit_test_results = _load_json(REPORTS / "unit-test-results.json")
    fault_data        = _load_json(REPORTS / f"fault-analysis-report-{target}.json")
    review_data       = _load_json(REPORTS / f"code-review-{target}.json")

    existing_summary  = _summarise_existing_tests(unit_test_results)
    fault_context     = _summarise_fault_context(fault_data)
    code_issues       = _summarise_code_issues(review_data)

    sources     = _load_source_files()
    source_text = "\n\n".join(
        f"=== {name} ===\n{content[:2000]}"
        for name, content in sources.items()
    )
    if not sources:
        source_text = "Source files not found — generating generic tests."

    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=3500,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert embedded systems test engineer for
ESP32 / ESP-IDF / ESP-Matter firmware using the Unity framework.
CRITICAL RULES — ALL must be followed:
1. ALL newlines inside JSON string values MUST be escaped as \\n
2. Respond with valid JSON only — no markdown, no backticks
3. NEVER use test_placeholder — it is a stub that tests NOTHING
4. Generate REAL test functions with actual TEST_ASSERT_* calls
5. test_file_content MUST implement every test in test_cases
6. Every test function must have at least one TEST_ASSERT_EQUAL or TEST_ASSERT_NOT_NULL
7. Use types from mock_idf.h: esp_err_t, ESP_OK, ESP_ERR_INVALID_ARG, etc.
"""),
        ("human", """Generate unit and integration tests for ESP-Matter light.
Target: {target}

=== EXISTING TEST STATUS ===
{existing_summary}

=== FAULT INJECTION CONTEXT (robustness regressions to cover) ===
{fault_context}

=== CODE REVIEW ISSUES (to add test coverage for) ===
{code_issues}

=== SOURCE CODE ===
{source_text}

Respond with exactly this JSON (escape ALL newlines in code as \\n):
{{
  "target": "{target}",
  "test_file_name": "test_light_app_{target}.cpp",
  "existing_coverage": "none",
  "test_file_content": "#include <unity.h>\\n#include \\"mock_idf.h\\"\\n\\nvoid setUp(void) {{}}\\nvoid tearDown(void) {{}}\\n\\n/* Test 1: on/off cluster — normal operation */\\nvoid test_on_off_cluster_toggle(void) {{\\n    esp_matter_attr_val_t val;\\n    val.type = ESP_MATTER_VAL_TYPE_BOOLEAN;\\n    val.val.b = true;\\n    esp_err_t ret = app_driver_attribute_update(NULL, 1, 0x0006, &val);\\n    TEST_ASSERT_EQUAL(ESP_OK, ret);\\n}}\\n\\n/* Test 2: NULL endpoint — robustness */\\nvoid test_null_endpoint_handled(void) {{\\n    esp_err_t ret = app_driver_attribute_update(NULL, 0, 0x0006, NULL);\\n    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, ret);\\n}}\\n\\n/* Test 3: invalid attribute type — boundary check */\\nvoid test_invalid_attribute_type(void) {{\\n    esp_matter_attr_val_t val;\\n    val.type = ESP_MATTER_VAL_TYPE_INVALID;\\n    esp_err_t ret = app_driver_attribute_update(NULL, 1, 0x0006, &val);\\n    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, ret);\\n}}\\n\\nvoid app_main(void) {{\\n    UNITY_BEGIN();\\n    RUN_TEST(test_on_off_cluster_toggle);\\n    RUN_TEST(test_null_endpoint_handled);\\n    RUN_TEST(test_invalid_attribute_type);\\n    UNITY_END();\\n}}",
  "mock_files": [
    {{"filename": "mock_led_driver.h", "content": "/* mock content \\\\n */"}},
    {{"filename": "mock_gpio.h",       "content": "/* mock content \\\\n */"}}
  ],
  "test_cases": [
    {{
      "name": "test_on_off_cluster_toggle",
      "type": "unit",
      "area": "on_off",
      "description": "Verifies on/off cluster toggle",
      "framework_call": "TEST_ASSERT_EQUAL(expected, actual)"
    }},
    {{
      "name": "test_null_endpoint_handled",
      "type": "robustness",
      "area": "matter_null_endpoint",
      "description": "NULL endpoint returns ESP_ERR_INVALID_STATE without crash",
      "framework_call": "TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, result)"
    }},
    {{
      "name": "test_malloc_exhaustion_graceful",
      "type": "robustness",
      "area": "memory",
      "description": "malloc() failure returns ESP_ERR_NO_MEM without panic",
      "framework_call": "TEST_ASSERT_EQUAL(ESP_ERR_NO_MEM, result)"
    }}
  ],
  "edge_cases_covered": [
    "NULL pointer handling",
    "Invalid attribute values",
    "Heap exhaustion (malloc returns NULL)",
    "NVS key not found — default values used",
    "Stack overflow detection via canary",
    "Matter attribute out-of-range clamping"
  ],
  "regression_tests": [
    {{
      "scenario": "null_pointer_deref",
      "test_name": "test_null_pointer_deref_regression",
      "description": "Ensures endpoint NULL is handled without panic after autofix patch"
    }}
  ],
  "sdk_integration_points": [
    {{
      "function": "esp_matter::attribute::update()",
      "test_approach": "Mock attribute to test update function"
    }}
  ],
  "run_instructions": "idf.py -C esp-matter/examples/light/test build -DTEST_BUILD=1",
  "summary": "Test suite for ESP-Matter light firmware on {target} — covers normal operation, robustness regressions, and security boundary checks"
}}"""),
    ])

    chain = prompt | llm | StrOutputParser()

    print("[Test Gen Agent] Calling Groq LLM...")
    raw = chain.invoke({
        "target":           target,
        "existing_summary": existing_summary,
        "fault_context":    fault_context,
        "code_issues":      code_issues,
        "source_text":      source_text,
    })

    # Robust parse — handles unescaped newlines in code strings
    report = _robust_parse(raw, target)

    if report.get("parse_error") and not report.get("parse_recovered"):
        print("[Test Gen Agent] Warning: JSON parse failed — raw saved")
    elif report.get("parse_recovered"):
        print("[Test Gen Agent] JSON recovered via manual extraction")
    else:
        print("[Test Gen Agent] JSON parsed successfully")

    REPORTS.mkdir(exist_ok=True)

    # ── Fallback: if LLM returned placeholder, generate real tests ──
    cpp_check = report.get("test_file_content", "")
    if "test_placeholder" in cpp_check and "TEST_PASS()" in cpp_check:
        print("[Test Gen Agent] LLM returned placeholder — generating real tests locally")
        real_cpp = (
            '#include <unity.h>\n'
            '#include "mock_idf.h"\n\n'
            'void setUp(void) {}\n'
            'void tearDown(void) {}\n\n'
            '/* Test: on/off attribute update */\n'
            'void test_on_off_attribute_update(void) {\n'
            '    esp_matter_attr_val_t val;\n'
            '    val.type = ESP_MATTER_VAL_TYPE_BOOLEAN;\n'
            '    val.val.b = true;\n'
            '    esp_err_t r = app_driver_attribute_update(NULL, 1, 0x0006, &val);\n'
            '    TEST_ASSERT_EQUAL(ESP_OK, r);\n'
            '}\n\n'
            '/* Test: NULL val pointer rejected */\n'
            'void test_null_val_rejected(void) {\n'
            '    esp_err_t r = app_driver_attribute_update(NULL, 1, 0x0006, NULL);\n'
            '    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, r);\n'
            '}\n\n'
            '/* Test: INVALID type rejected */\n'
            'void test_invalid_type_rejected(void) {\n'
            '    esp_matter_attr_val_t val;\n'
            '    val.type = ESP_MATTER_VAL_TYPE_INVALID;\n'
            '    esp_err_t r = app_driver_attribute_update(NULL, 1, 0x0006, &val);\n'
            '    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, r);\n'
            '}\n\n'
            '/* Test: brightness range 0-254 */\n'
            'void test_brightness_boundary(void) {\n'
            '    esp_matter_attr_val_t val;\n'
            '    val.type = ESP_MATTER_VAL_TYPE_INTEGER;\n'
            '    val.val.i = 254;\n'
            '    esp_err_t r = app_driver_attribute_update(NULL, 1, 0x0008, &val);\n'
            '    TEST_ASSERT_EQUAL(ESP_OK, r);\n'
            '}\n\n'
            'void app_main(void) {\n'
            '    UNITY_BEGIN();\n'
            '    RUN_TEST(test_on_off_attribute_update);\n'
            '    RUN_TEST(test_null_val_rejected);\n'
            '    RUN_TEST(test_invalid_type_rejected);\n'
            '    RUN_TEST(test_brightness_boundary);\n'
            '    UNITY_END();\n'
            '}\n'
        )
        report["test_file_content"] = real_cpp
        report["fallback_used"] = True
        print(f"[Test Gen Agent] Real tests generated locally ({len(real_cpp)} chars)")

    # ── Save JSON report ──────────────────────────────────────────
    out_json = REPORTS / f"testgen-report-{target}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── Save generated C++ test file to reports/ (artifact upload) ─
    cpp_content = report.get("test_file_content", "")
    if cpp_content:
        out_cpp = REPORTS / f"generated_tests_{target}.cpp"
        out_cpp.write_text(cpp_content, encoding="utf-8")
        print(f"[Test Gen Agent] C++ test file saved: {out_cpp} ({len(cpp_content)} chars)")
    else:
        print("[Test Gen Agent] Warning: no test_file_content in report")

    # ── Save mock files to reports/ ───────────────────────────────
    for mock in report.get("mock_files", []):
        fn   = mock.get("filename", "")
        cont = mock.get("content", "")
        if fn and cont:
            (REPORTS / fn).write_text(cont, encoding="utf-8")
            print(f"[Test Gen Agent] Mock saved: {fn}")

    # ── DEPLOY to ESP-Matter test directory (NEW) ─────────────────
    deploy_manifest = {"status": "skipped", "reason": "no test_file_content"}
    if cpp_content:
        print("[Test Gen Agent] Deploying test file to ESP-Matter test dir...")
        deploy_manifest = _deploy_test_file(
            cpp_content=cpp_content,
            target=target,
            mock_files=report.get("mock_files", []),
        )
        print(f"[Test Gen Agent] Deploy status: {deploy_manifest['status']}")
        if deploy_manifest.get("deployed_files"):
            for f in deploy_manifest["deployed_files"]:
                print(f"  → {f}")

    report["deploy_manifest"] = deploy_manifest
    # Re-save with deploy info included
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    n_tests = len(report.get("test_cases", []))
    n_regressions = len(report.get("regression_tests", []))
    print(f"[Test Gen Agent] Done — {n_tests} test cases, {n_regressions} regression tests")
    print(f"[Test Gen Agent] Report: {out_json}")
    return report


if __name__ == "__main__":
    report = run_test_gen_agent()
    print(json.dumps({
        k: v for k, v in report.items()
        if k not in ("test_file_content",)
    }, indent=2))