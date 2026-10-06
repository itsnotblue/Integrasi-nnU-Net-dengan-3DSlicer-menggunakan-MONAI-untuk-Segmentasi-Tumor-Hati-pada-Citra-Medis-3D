"""Paired case-bootstrap differences for two complete, comparable evaluations."""

import argparse
from datetime import datetime, timezone

from evaluation_common import read_json, sha256, write_json_new
from evaluate_predictions import describe


def compare(left, right, samples=2000, seed=20261006):
    if left["manifest_sha256"] != right["manifest_sha256"]:
        raise ValueError("Comparisons must use the same locked cohort manifest")
    if left["protocol"] != right["protocol"]:
        raise ValueError("Metric protocols differ")
    left_cases = {case["case_id"]: case for case in left["per_case"]}
    right_cases = {case["case_id"]: case for case in right["per_case"]}
    if len(left_cases) != len(left["per_case"]) or len(right_cases) != len(right["per_case"]) or set(left_cases) != set(right_cases):
        raise ValueError("Case memberships differ or contain duplicates")
    ids = sorted(left_cases)
    differences = {}
    for name in left_cases[ids[0]]["classes"]:
        differences[name] = {}
        for metric in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm"):
            values = []
            for case_id in ids:
                first = left_cases[case_id]["classes"][name][metric]
                second = right_cases[case_id]["classes"][name][metric]
                values.append(first - second if first is not None and second is not None else None)
            differences[name][metric] = describe(values, samples, seed)
    return {"left_model": left["model"], "right_model": right["model"],
            "difference_direction": "left minus right; positive overlap/recall/precision favors left, negative distance favors left",
            "left_checkpoint_sha256": left["checkpoint_sha256"], "right_checkpoint_sha256": right["checkpoint_sha256"],
            "left_checkpoint_attribution": left.get("checkpoint_attribution", "unverified_legacy_report"),
            "right_checkpoint_attribution": right.get("checkpoint_attribution", "unverified_legacy_report"),
            "cohort_id": left["cohort_id"], "case_ids": ids, "held_out_certified": False,
            "bootstrap": {"samples": samples, "seed": seed, "unit": "paired case", "method": "percentile 95% CI of mean difference"},
            "paired_differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap replicates")
    result = compare(read_json(args.left), read_json(args.right), args.bootstrap_samples, args.seed)
    result.update(schema_version=1, analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
                  left_evaluation_sha256=sha256(args.left), right_evaluation_sha256=sha256(args.right))
    write_json_new(args.output, result)
    print(f"Saved paired comparison: {args.output}")


if __name__ == "__main__":
    main()
