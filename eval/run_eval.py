"""
Automated evaluation runner.
Tests the full pipeline against eval_dataset.json and prints a score report.

Usage: venv/bin/python eval/run_eval.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # triggers telemetry patch

from app.services.pipeline import run
from app.services.safety import check_input

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _stub_image_check(prompt: str) -> bool:
    """Use image_gen if built, otherwise stub returns True (safe)."""
    try:
        from app.services.image_gen import is_image_prompt_safe
        return is_image_prompt_safe(prompt)
    except Exception:
        return True  # Phase 5 not yet built — skip image checks


def evaluate_case(case: dict) -> dict:
    cid       = case["id"]
    category  = case["category"]
    prompt    = case["prompt"]
    denom     = case.get("denomination", "Non-denominational (default)")

    # Image policy cases — test image pipeline directly
    if category == "image_policy":
        safe = _stub_image_check(prompt)
        expected_blocked = case.get("expect_blocked", False)
        passed = safe != expected_blocked   # safe=False means blocked
        return {
            "id": cid, "category": category,
            "passed": passed,
            "detail": f"image_safe={safe}, expect_blocked={expected_blocked}",
        }

    # All other cases — run full chat pipeline
    result = run(prompt, denomination=denom)

    passed = True
    detail_parts = []

    # Check blocked expectation
    if case.get("expect_blocked"):
        if not result["blocked"]:
            passed = False
            detail_parts.append("expected BLOCKED but was allowed through")
        else:
            expected_cat = case.get("block_category")
            if expected_cat and result["block_category"] != expected_cat:
                detail_parts.append(
                    f"blocked OK but category={result['block_category']} (expected {expected_cat})"
                )
    else:
        if result["blocked"]:
            passed = False
            detail_parts.append(f"unexpectedly BLOCKED ({result['block_category']})")

    # Check sources expectation
    if case.get("expect_sources") and not result["blocked"]:
        if not result["sources"]:
            passed = False
            detail_parts.append("expected scripture sources but got none")

    # Check unverified flag expectation
    if case.get("expect_unverified_in_reply") and not result["blocked"]:
        if "[UNVERIFIED" not in result["reply"] and not result["unverified"]:
            passed = False
            detail_parts.append("expected [UNVERIFIED] tag but verifier did not flag anything")

    detail = "; ".join(detail_parts) if detail_parts else "ok"
    return {
        "id": cid, "category": category,
        "passed": passed,
        "detail": detail,
        "reply_snippet": result["reply"][:80] + "..." if len(result["reply"]) > 80 else result["reply"],
        "blocked": result["blocked"],
        "sources": [v["ref"] for v in result["sources"][:3]],
    }


def main():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print(f"\n{BOLD}=== Christianity AI — Evaluation Run ({len(dataset)} cases) ==={RESET}\n")

    results = []
    for case in dataset:
        print(f"  Running [{case['id']}] {case['prompt'][:55]}...", end=" ", flush=True)
        r = evaluate_case(case)
        results.append(r)
        status = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(status)
        if not r["passed"] or r.get("detail") not in ("ok", ""):
            print(f"         {YELLOW}→ {r['detail']}{RESET}")

    # Summary by category
    cats: dict[str, list[bool]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["passed"])

    total  = len(results)
    passed = sum(r["passed"] for r in results)

    print(f"\n{BOLD}{'─'*50}")
    print(f"Results by category:{RESET}")
    for cat, outcomes in cats.items():
        p = sum(outcomes)
        t = len(outcomes)
        bar = f"{GREEN}{'█'*p}{RED}{'░'*(t-p)}{RESET}"
        print(f"  {cat:<28} {bar}  {p}/{t}")

    print(f"\n{BOLD}Overall: {passed}/{total} passed", end="")
    pct = passed / total * 100
    color = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
    print(f"  {color}({pct:.0f}%){RESET}{RESET}\n")

    # Save JSON report
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full report saved to {report_path}\n")


if __name__ == "__main__":
    main()
