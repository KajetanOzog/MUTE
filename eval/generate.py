                      
import argparse
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    REPO_ROOT,
    Model,
    candidate_inputs,
    load_config,
    read_jsonl,
    shard,
    slugify,
    write_jsonl,
)
from datasets import load_train_records


def completed_records(path):
    if not path.exists():
        return {}
    return {
        record["id"]: record
        for record in read_jsonl(path)
        if record.get("response")
    }


def load_candidates(path):
    candidates = read_jsonl(path)
    seen = set()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.get("id")
        prompt = candidate.get("text") or candidate.get("prompt")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{path}:{index}: candidate requires a non-empty id")
        if candidate_id in seen:
            raise ValueError(f"{path}:{index}: duplicate candidate id {candidate_id!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{index}: candidate requires a non-empty prompt")
        seen.add(candidate_id)
    return candidates


def expand_candidates(records, candidates):
    expanded = []
    for candidate in candidates:
        candidate_prompt = candidate.get("text") or candidate.get("prompt")
        for source in records:
            record = dict(source)
            record["source_id"] = source["id"]
            record["id"] = f"{candidate['id']}__{source['id']}"
            record["candidate_id"] = candidate["id"]
            record["candidate_prompt"] = candidate_prompt.strip()
            record["candidate_brand"] = candidate.get("brand")
            for key in (
                "generation",
                "parent_ids",
                "generator_seed",
                "generation_attempt",
                "source_index",
                "generator_model",
            ):
                if key in candidate:
                    record[key] = candidate[key]
            expanded.append(record)
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--retain-dataset")
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.model_entry(args.model)
    try:
        records = load_train_records(
            config,
            brand=args.brand,
            retain_dataset=args.retain_dataset,
        )
        if args.limit:
            records = records[: args.limit]
        if args.candidate_file:
            records = expand_candidates(
                records, load_candidates(args.candidate_file)
            )
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"cannot load evaluation data: {error}") from error

    records = shard(records, args.num_shards, args.shard_id)
    run_name = args.run_name or "responses"
    output = Path(args.out) / run_name / f"shard_{args.shard_id}.jsonl"
    completed = {} if args.no_resume else completed_records(output)
    if any(record.get("candidate_role", "user") != "system"
           for record in completed.values()):
        raise ValueError("Cannot resume responses generated with a different candidate role")
    pending = [record for record in records if record["id"] not in completed]

    print(
        f"[generate] {args.model}: {len(pending)}/{len(records)} records "
        f"in shard {args.shard_id}/{args.num_shards}"
    )
    if not pending:
        return

    model = Model(
        model_config["path"],
        runtime=config.runtime,
        model_config=model_config,
        tokenizer_path=model_config.get("tokenizer"),
    )
    prompts, systems = candidate_inputs(pending)
    responses = model.generate(prompts, systems=systems)
    for record, response in zip(pending, responses):
        record["response"] = response
        record["model"] = args.model
        record["data_split"] = args.split
        record["candidate_role"] = "system"

    write_jsonl(
        output,
        [completed.get(record["id"], record) for record in records],
    )
    print(f"[generate] wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
