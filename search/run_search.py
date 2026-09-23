                      
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "eval"
sys.path.insert(0, str(EVAL_ROOT))

from common import DEFAULT_CONFIG, load_config, read_jsonl, slugify, portable_path
from datasets import discover_train_forget


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else config.path.parent / path


def resolve_meta_prompt(config, mode, explicit, brand):
    if explicit:
        return Path(explicit).resolve()
    meta = config.search["meta_prompts"]
    root = resolve_config_path(config, meta["directory"])
    canonical = config.canonical(slugify(brand))
    relative = meta[mode].format(
        brand=canonical,
        brand_slug=slugify(canonical),
        category=config.brands[canonical]["category"],
    )
    return (root / relative).resolve()


def run_command(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("[search] exec:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def merge_candidates(existing, new):
    by_id = {candidate["id"]: candidate for candidate in existing}
    for candidate in new:
        by_id[candidate["id"]] = candidate
    return sorted(
        by_id.values(),
        key=lambda candidate: (
            candidate.get("generation", 0),
            candidate["id"],
        ),
    )


def ranked(candidates):
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.get("fitness") is not None
        ),
        key=lambda candidate: candidate["fitness"],
        reverse=True,
    )


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare_retain_sample(config, run_dir):
    dataset_config = config.search["datasets"]
    source_path = resolve_config_path(
        config, dataset_config["train_retain"]
    ).resolve()
    sample_size = int(dataset_config["retain_sample_size"])
    sample_seed = int(dataset_config["retain_sample_seed"])
    if sample_size <= 0:
        raise ValueError("retain_sample_size must be positive")

    snapshot_dir = run_dir / "snapshot"
    sample_path = snapshot_dir / "train_retain_sample.jsonl"
    selection_path = snapshot_dir / "retain_sample_selection.json"
    if sample_path.exists() != selection_path.exists():
        raise ValueError("incomplete retain sample checkpoint")
    if sample_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected = {
            "source": portable_path(source_path),
            "sample_size": sample_size,
            "sample_seed": sample_seed,
        }
        for key, value in expected.items():
            if selection.get(key) != value:
                raise ValueError(
                    f"retain sample checkpoint differs for {key}: "
                    f"{selection.get(key)!r} != {value!r}"
                )
        if len(read_jsonl(sample_path)) != sample_size:
            raise ValueError("retain sample checkpoint has an invalid row count")
        return source_path, sample_path, selection

    records = read_jsonl(source_path)
    if sample_size > len(records):
        raise ValueError(
            f"retain sample requests {sample_size} of only {len(records)} records"
        )
    selected_indices = sorted(
        random.Random(sample_seed).sample(range(len(records)), sample_size)
    )
    selected_records = [records[index] for index in selected_indices]
    selection = {
        "source": portable_path(source_path),
        "source_sha256": sha256(source_path),
        "source_count": len(records),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "source_rows_1_based": [index + 1 for index in selected_indices],
    }
    atomic_jsonl(sample_path, selected_records)
    atomic_json(selection_path, selection)
    return source_path, sample_path, selection


def make_manifest(
    args, config, run_dir, initial_meta, mutation_meta,
    retain_source, retain_sample, retain_selection,
):
    canonical, forget_path = discover_train_forget(
        REPO_ROOT / "dataset", config, args.brand
    )
    immutable = {
        "brand": canonical,
        "generator_model": config.generator["model"],
        "generator_model_config": config.model_entry(config.generator["model"]),
        "generator_parameters": config.generator["runtime"],
        "generator_seed": config.search["generator_seed"],
        "target_model": args.target_model,
        "target_model_config": config.model_entry(args.target_model),
        "target_parameters": config.runtime,
        "judge_model": config.judge["model"],
        "judge_model_config": config.model_entry(config.judge["model"]),
        "judge_parameters": config.judge["runtime"],
        "search": {
            **({"candidate_role": config.search["candidate_role"]}
               if "candidate_role" in config.search else {}),
            **{key: config.search[key]
            for key in (
                "seed",
                "num_candidates",
                "prompts_per_call",
                "max_generation_attempts",
                "allow_partial_population",
                "top_k",
                "mutation_rounds",
                "candidate_input_template",
            )}
        },
        "datasets": {
            "forget": portable_path(forget_path.resolve()),
            "forget_sha256": sha256(forget_path),
            "retain_source": portable_path(retain_source),
            "retain_source_sha256": sha256(retain_source),
            "retain_sample": portable_path(retain_sample),
            "retain_sample_sha256": sha256(retain_sample),
            "retain_sample_size": retain_selection["sample_size"],
            "retain_sample_seed": retain_selection["sample_seed"],
        },
        "meta_prompts": {
            "initial": portable_path(initial_meta),
            "initial_sha256": sha256(initial_meta),
            "mutation": portable_path(mutation_meta),
            "mutation_sha256": sha256(mutation_meta),
        },
        "config_sha256": sha256(config.path),
        "runtime_environment": {
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in ("torch", "transformers", "vllm")
            },
        },
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("immutable") != immutable:
            raise SystemExit(
                "resume configuration differs from manifest; use a new run directory"
            )
        return manifest

    snapshot_dir = run_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.path, snapshot_dir / "config.yaml")
    shutil.copy2(initial_meta, snapshot_dir / "initial_meta_prompt.txt")
    shutil.copy2(mutation_meta, snapshot_dir / "mutation_meta_prompt.txt")
    manifest = {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "immutable": immutable,
        "phases": {},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def update_manifest(run_dir, manifest, **updates):
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    atomic_json(run_dir / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--target-model")
    parser.add_argument("--run-dir")
    parser.add_argument("--initial-meta-prompt")
    parser.add_argument("--mutation-meta-prompt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    args.target_model = args.target_model or config.search.get("target_model")
    if not args.target_model:
        raise SystemExit(
            "target model is not configured; pass --target-model or set "
            "search.target_model"
        )
    config.model_entry(args.target_model)
    if config.search.get("candidate_role") != "system":
        raise ValueError("MUTE requires candidate_role: system")

    args.brand = config.canonical(slugify(args.brand))

    initial_meta = resolve_meta_prompt(
        config, "initial", args.initial_meta_prompt, args.brand
    )
    mutation_meta = resolve_meta_prompt(
        config, "mutation", args.mutation_meta_prompt, args.brand
    )
    for path in (initial_meta, mutation_meta):
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise SystemExit(f"missing or empty meta-prompt: {path}")

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            REPO_ROOT / "runs" / slugify(args.brand) / stamp
        ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[search] run directory: {run_dir}")

    try:
        retain_source, retain_sample, retain_selection = prepare_retain_sample(
            config, run_dir
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot prepare retain sample: {error}") from error
    manifest = make_manifest(
        args, config, run_dir, initial_meta, mutation_meta,
        retain_source, retain_sample, retain_selection,
    )
    pool_path = run_dir / "candidate_pool.jsonl"
    ranking_path = run_dir / "global_ranking.jsonl"
    pool = read_jsonl(pool_path) if pool_path.exists() else []

    total_generations = config.search["mutation_rounds"] + 1
    for generation in range(total_generations):
        generation_dir = run_dir / f"generation_{generation:02d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = generation_dir / "candidates.jsonl"
        scores_path = generation_dir / "scores.jsonl"
        checkpoint_path = generation_dir / "evaluation_complete.json"

        if scores_path.exists():
            generation_scores = read_jsonl(scores_path)
            if not generation_scores:
                raise SystemExit(f"empty score checkpoint: {scores_path}")
            pool = merge_candidates(pool, generation_scores)
            global_ranking = ranked(pool)
            if not global_ranking:
                raise SystemExit(
                    f"completed generation {generation} has no rankable candidates"
                )
            atomic_jsonl(pool_path, pool)
            atomic_jsonl(ranking_path, global_ranking)
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            else:
                checkpoint = {
                    "generation": generation,
                    "candidate_count_this_generation": len(generation_scores),
                    "candidate_count_global": len(pool),
                    "rankable_count_global": len(global_ranking),
                    "best_candidate_id": global_ranking[0]["id"],
                    "best_fitness": global_ranking[0]["fitness"],
                    "completed_at": utc_now(),
                    "recovered_from_scores": True,
                }
                atomic_json(checkpoint_path, checkpoint)
            atomic_jsonl(
                generation_dir / "top_k.jsonl",
                global_ranking[: config.search["top_k"]],
            )
            manifest["phases"][f"generation_{generation:02d}"] = checkpoint
            update_manifest(run_dir, manifest, status="running")
            print(f"[search] resume: generation {generation} already complete")
            continue

        generator_args = [
            "search/generate_candidates.py",
            "--mode", "initial" if generation == 0 else "mutation",
            "--brand", args.brand,
            "--generation", generation,
            "--candidate-offset", len(pool),
            "--run-dir", generation_dir,
            "--config", config.path,
            "--meta-prompt",
            initial_meta if generation == 0 else mutation_meta,
        ]
        if generation > 0:
            generator_args.extend(
                [
                    "--parents", ranking_path,
                    "--existing-candidates", pool_path,
                ]
            )
        run_command(generator_args)

        target_root = generation_dir / "target_runs"
        target_run = target_root / "responses"
        run_command(
            [
                "eval/generate.py",
                "--model", args.target_model,
                "--split", "train",
                "--brand", args.brand,
                "--retain-dataset", retain_sample,
                "--candidate-file", candidates_path,
                "--run-name", "responses",
                "--out", target_root,
                "--config", config.path,
            ]
        )

        judged_root = generation_dir / "judged"
        judged_run = judged_root / "responses"
        run_command(
            [
                "eval/judge.py",
                "--run", target_run,
                "--out", judged_root,
                "--include-quality",
                "--config", config.path,
            ]
        )

        run_command(
            [
                "eval/search_metrics.py",
                "--judged-run", judged_run,
                "--out", scores_path,
            ]
        )

        pool = merge_candidates(pool, read_jsonl(scores_path))
        global_ranking = ranked(pool)
        if not global_ranking:
            raise SystemExit(
                f"generation {generation} produced no rankable candidates"
            )
        atomic_jsonl(pool_path, pool)
        atomic_jsonl(ranking_path, global_ranking)
        atomic_jsonl(
            generation_dir / "top_k.jsonl",
            global_ranking[: config.search["top_k"]],
        )
        checkpoint = {
            "generation": generation,
            "candidate_count_this_generation": len(read_jsonl(scores_path)),
            "candidate_count_global": len(pool),
            "rankable_count_global": len(global_ranking),
            "best_candidate_id": global_ranking[0]["id"],
            "best_fitness": global_ranking[0]["fitness"],
            "completed_at": utc_now(),
        }
        atomic_json(generation_dir / "evaluation_complete.json", checkpoint)
        manifest["phases"][f"generation_{generation:02d}"] = checkpoint
        update_manifest(run_dir, manifest, status="running")

    global_ranking = ranked(pool)
    best = global_ranking[0]
    atomic_jsonl(run_dir / "best_candidate.jsonl", [best])
    (run_dir / "best_prompt.txt").write_text(
        best["text"].rstrip() + "\n", encoding="utf-8"
    )
    update_manifest(
        run_dir,
        manifest,
        status="complete",
        completed_at=utc_now(),
        best_candidate_id=best["id"],
        best_fitness=best["fitness"],
        candidate_count=len(pool),
    )
    print(
        f"[search] complete: best={best['id']} fitness={best['fitness']:.6f}"
    )


if __name__ == "__main__":
    main()
