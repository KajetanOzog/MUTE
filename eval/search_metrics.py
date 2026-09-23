                      
import argparse
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, load_shards, write_jsonl


def forget_leak(judgment):
    brand = judgment.get("target_brand_present")
    trade_dress = judgment.get("target_trade_dress_present")
    if brand is True or trade_dress is True:
        return 1.0
    if brand is False and trade_dress is False:
        return 0.0
    return None


def retain_value(judgment):
    value = judgment.get("qa_correct")
    return float(value) if isinstance(value, bool) else None


def quality_value(judgment):
    value = judgment.get("quality_1_5")
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
        return float(value)
    return None


def harmonic_fitness(forget_score, retain_score):
    if forget_score is None or retain_score is None:
        return None
    forget_success = 1.0 - forget_score
    total = forget_success + retain_score
    if total == 0:
        return 0.0
    return 2.0 * forget_success * retain_score / total


def _summary(values, invalid):
    return {
        "value": sum(values) / len(values) if values else None,
        "n_valid": len(values),
        "n_invalid": invalid,
    }


def score_candidates(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("candidate_id", "__single__")].append(record)

    results = []
    for candidate_id, candidate_records in grouped.items():
        forget_values = []
        retain_values = []
        quality_values = []
        invalid = {"forget": 0, "retain": 0, "quality": 0}

        for record in candidate_records:
            judgment = record.get("judgment") or {}
            task = record.get("task")
            if task == "forget":
                value = forget_leak(judgment)
                if value is None:
                    invalid["forget"] += 1
                else:
                    forget_values.append(value)
            elif task == "retain":
                value = retain_value(judgment)
                if value is None:
                    invalid["retain"] += 1
                else:
                    retain_values.append(value)

            quality = quality_value(judgment)
            if quality is None:
                invalid["quality"] += 1
            else:
                quality_values.append(quality)

        forget = _summary(forget_values, invalid["forget"])
        retain = _summary(retain_values, invalid["retain"])
        quality = _summary(quality_values, invalid["quality"])
        fitness = harmonic_fitness(forget["value"], retain["value"])
        first = candidate_records[0]
        any_invalid = any(invalid.values())
        if fitness is None:
            evaluation_status = "failed"
            error = "missing valid forget_score or retain_score"
        elif quality["value"] is None or any_invalid:
            evaluation_status = "complete_with_invalid"
            error = None
        else:
            evaluation_status = "complete"
            error = None
        results.append(
            {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "brand": first.get("candidate_brand"),
                "text": first.get("candidate_prompt"),
                "candidate_prompt": first.get("candidate_prompt"),
                "generation": first.get("generation"),
                "parent_ids": first.get("parent_ids") or [],
                "generator_seed": first.get("generator_seed"),
                "generation_attempt": first.get("generation_attempt"),
                "source_index": first.get("source_index"),
                "generator_model": first.get("generator_model"),
                "scores": {
                    "forget": forget["value"],
                    "retain": retain["value"],
                    "quality": quality["value"],
                    "fitness": fitness,
                },
                "successes": {
                    "forget": (
                        1.0 - forget["value"]
                        if forget["value"] is not None
                        else None
                    ),
                    "retain": retain["value"],
                },
                "forget_score": forget["value"],
                "retain_score": retain["value"],
                "forget_success": (
                    1.0 - forget["value"]
                    if forget["value"] is not None
                    else None
                ),
                "retain_success": retain["value"],
                "quality_score": quality["value"],
                "fitness": fitness,
                "status": evaluation_status,
                "evaluation_status": evaluation_status,
                "error": error,
                "counts": {
                    "forget": {
                        "n_valid": forget["n_valid"],
                        "n_invalid": forget["n_invalid"],
                    },
                    "retain": {
                        "n_valid": retain["n_valid"],
                        "n_invalid": retain["n_invalid"],
                    },
                    "quality": {
                        "n_valid": quality["n_valid"],
                        "n_invalid": quality["n_invalid"],
                    },
                },
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item["fitness"] is not None,
            item["fitness"] if item["fitness"] is not None else -1.0,
        ),
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged-run", required=True)
    parser.add_argument("--out")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.judged_run)
    records = load_shards(run_dir)
    if not records:
        raise SystemExit(f"no judged shards found under {run_dir}")

    scores = score_candidates(records)
    if args.top_k:
        scores = scores[: args.top_k]
    output = Path(args.out) if args.out else (
        REPO_ROOT / "search_results" / run_dir.name / "candidate_scores.jsonl"
    )
    write_jsonl(output, scores)
    print(f"[search-metrics] wrote {len(scores)} candidates to {output}")


if __name__ == "__main__":
    main()
