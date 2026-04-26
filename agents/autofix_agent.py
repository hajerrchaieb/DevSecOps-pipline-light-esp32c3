import subprocess
import os
from datetime import datetime
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
TARGET = os.getenv("TARGET_CHIP", "esp32c3")
PATCHES = Path("reports/patches")

BRANCH_NAME = f"autofix/{TARGET}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# ==========================================
# STEP 1 — APPLY PATCHES
# ==========================================
def apply_patches():
    patch_files = list(PATCHES.glob("*.patch"))

    if not patch_files:
        print("[AutoFix] No patches found")
        return False

    applied = False

    for pf in patch_files:
        print(f"[AutoFix] Applying {pf.name}")
        result = subprocess.run(
            ["git", "apply", str(pf)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  ✅ Applied {pf.name}")
            applied = True
        else:
            print(f"  ⚠️ Skipped {pf.name}")
    
    return applied


# ==========================================
# STEP 2 — CREATE BRANCH + COMMIT
# ==========================================
def commit_changes():
    print("[AutoFix] Creating branch...")

    subprocess.run(["git", "checkout", "-b", BRANCH_NAME], check=True)

    subprocess.run(["git", "add", "."], check=True)

    subprocess.run([
        "git", "commit", "-m",
        f"🤖 AutoFix: apply patches for {TARGET}"
    ], check=True)

    print("[AutoFix] Commit created")


# ==========================================
# STEP 3 — PUSH BRANCH
# ==========================================
def push_branch():
    print("[AutoFix] Pushing branch...")

    subprocess.run([
        "git", "push", "origin", BRANCH_NAME
    ], check=True)

    print("[AutoFix] Branch pushed")


# ==========================================
# STEP 4 — CREATE PULL REQUEST
# ==========================================
def create_pr():
    print("[AutoFix] Creating Pull Request...")

    title = f"🤖 AutoFix ({TARGET})"
    body = f"""
Auto-generated fixes by AutoFix Agent

Target: {TARGET}
Time: {datetime.now()}

Includes:
- Bug fixes
- Security fixes
- Code quality improvements
"""

    subprocess.run([
        "gh", "pr", "create",
        "--title", title,
        "--body", body,
        "--base", "main",
        "--head", BRANCH_NAME
    ], check=True)

    print("[AutoFix] PR created successfully 🎉")


# ==========================================
# MAIN
# ==========================================
def run():
    print("\n========== AutoFix PR Agent ==========")

    applied = apply_patches()

    if not applied:
        print("[AutoFix] No changes → No PR created")
        return

    commit_changes()
    push_branch()
    create_pr()

    print("\n✅ DONE — Pull Request created")


if __name__ == "__main__":
    run()
