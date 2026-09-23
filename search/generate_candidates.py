                      
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

from common import DEFAULT_CONFIG, Model, load_config, read_jsonl, slugify, write_jsonl


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render_template(template, values):
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", str(value))
    return rendered


def validate_template(template, mode):
    required = {"brand", "batch_size"}
    if mode == "mutation":
        required.update({"seed_prompts", "existing_prompts"})
    missing = sorted(
        placeholder
        for placeholder in required
        if "{" + placeholder + "}" not in template
    )
    if missing:
        raise ValueError(
            f"{mode} meta-prompt is missing required placeholders: "
            + ", ".join("{" + name + "}" for name in missing)
        )


def generation_call_seed(base_seed, attempt, generation, mode):
    seed = base_seed + attempt - 1
    if mode == "mutation":
        seed += generation * 1000
    return seed


def extract_first_json_container(text):
    for start_index, opening in enumerate(text):
        if opening not in "[{":
            continue
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start_index:index + 1]
    return None


def strip_thinking_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def clean_prompt_text(text):
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def parse_generator_response(text):
    container = extract_first_json_container(strip_thinking_tags(text))
    if not container:
        return []
    try:
        payload = json.loads(container)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        prompts = payload.get("prompts", [])
    elif isinstance(payload, list):
        prompts = payload
    else:
        prompts = []

    cleaned = []
    for item in prompts:
        if isinstance(item, str):
            value = clean_prompt_text(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            value = clean_prompt_text(item["prompt"])
        else:
            continue
        if value:
            cleaned.append(value)
    return cleaned


def generator_json_schema(count):
    return {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["prompts"],
        "additionalProperties": False,
    }


def dedupe_key(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def default_meta_prompt(config, mode, brand):
    meta = config.search["meta_prompts"]
    root = Path(meta["directory"])
    if not root.is_absolute():
        root = config.path.parent / root
    canonical = config.canonical(slugify(brand))
    relative = meta[mode].format(
        brand=canonical,
        brand_slug=slugify(canonical),
        category=config.brands[canonical]["category"],
    )
    return root / relative


def candidate_text(candidate):
    return candidate.get("text") or candidate.get("candidate_prompt") or candidate.get("prompt")


def load_ranked_parents(path, top_k):
    candidates = [
        candidate
        for candidate in read_jsonl(path)
        if candidate.get("fitness") is not None and candidate_text(candidate)
    ]
    candidates.sort(key=lambda item: item["fitness"], reverse=True)
    parents = candidates[:top_k]
    if len(parents) < top_k:
        raise ValueError(f"expected at least {top_k} scored parents, got {len(parents)}")
    return parents


def load_existing(path):
    if not path:
        return []
    return read_jsonl(path)


def attempt_numbers(attempts_dir):
    numbers = []
    for path in attempts_dir.glob("attempt_*.json"):
        match = re.fullmatch(r"attempt_(\d+)\.json", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


def build_instruction(template, brand, batch_size, top_k, parents, existing):
    parent_prompts = [candidate_text(parent) for parent in parents]
    existing_prompts = [candidate_text(candidate) for candidate in existing]
    values = {
        "brand": brand,
        "batch_size": batch_size,
        "count": batch_size,
        "top_k": top_k,
        "seed_prompts": json.dumps(parent_prompts, ensure_ascii=False),
        "parents": json.dumps(parent_prompts, ensure_ascii=False),
        "existing_prompts": json.dumps(existing_prompts, ensure_ascii=False),
    }
    return render_template(template, values)


def complete_candidate(
    candidate_id,
    brand,
    text,
    generation,
    parent_ids,
    call_seed,
    attempt,
    source_index,
    generator_model,
):
    return {
        "id": candidate_id,
        "brand": brand,
        "text": text,
        "prompt": text,
        "generation": generation,
        "parent_ids": list(parent_ids),
        "generator_seed": call_seed,
        "generation_attempt": attempt,
        "source_index": source_index,
        "generator_model": generator_model,
        "scores": {
            "forget": None,
            "retain": None,
            "quality": None,
            "fitness": None,
        },
        "successes": {"forget": None, "retain": None},
        "forget_score": None,
        "retain_score": None,
        "forget_success": None,
        "retain_success": None,
        "fitness": None,
        "quality_score": None,
        "status": "pending",
        "evaluation_status": "pending",
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "mutation"), required=True)
    parser.add_argument("--brand", required=True)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--parents")
    parser.add_argument("--existing-candidates")
    parser.add_argument("--meta-prompt")
    parser.add_argument("--count", type=int)
    parser.add_argument("--prompts-per-call", type=int)
    parser.add_argument("--max-generation-attempts", type=int)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.generator:
        raise SystemExit("generator is not configured")
    if args.mode == "mutation" and not args.parents:
        parser.error("--parents is required in mutation mode")

    generation = args.generation
    if generation is None:
        generation = 0 if args.mode == "initial" else 1
    count = args.count or config.search["num_candidates"]
    prompts_per_call = (
        args.prompts_per_call or config.search["prompts_per_call"]
    )
    max_attempts = (
        args.max_generation_attempts
        or config.search["max_generation_attempts"]
    )
    allow_partial = config.search.get("allow_partial_population", True)
    top_k = config.search["top_k"]
    generator_seed = config.search["generator_seed"]

    run_dir = Path(args.run_dir)
    attempts_dir = run_dir / "generator_attempts"
    accepted_path = run_dir / "accepted_candidates.jsonl"
    output_path = run_dir / "candidates.jsonl"
    complete_path = run_dir / "population_complete.json"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    if complete_path.exists() and output_path.exists() and not args.no_resume:
        candidates = read_jsonl(output_path)
        print(f"[generator] resume: population already complete ({len(candidates)})")
        return

    parents = load_ranked_parents(args.parents, top_k) if args.parents else []
    parent_ids = [parent.get("id") or parent.get("candidate_id") for parent in parents]
    existing = load_existing(args.existing_candidates)
    accepted = (
        read_jsonl(accepted_path)
        if accepted_path.exists() and not args.no_resume
        else []
    )
    seen = {
        dedupe_key(text)
        for text in (candidate_text(candidate) for candidate in existing + accepted)
        if text
    }

    meta_path = (
        Path(args.meta_prompt)
        if args.meta_prompt
        else default_meta_prompt(config, args.mode, args.brand)
    )
    try:
        template = meta_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise SystemExit(f"cannot load meta-prompt {meta_path}: {error}") from error
    if not template:
        raise SystemExit(f"meta-prompt is empty: {meta_path}")
    try:
        validate_template(template, args.mode)
    except ValueError as error:
        raise SystemExit(f"invalid meta-prompt {meta_path}: {error}") from error

    previous_attempts = attempt_numbers(attempts_dir) if not args.no_resume else []
    start_attempt = max(previous_attempts, default=0) + 1
    generator = config.generator
    model_config = config.model_entry(generator["model"])
    model = None

    for attempt in range(start_attempt, max_attempts + 1):
        if len(accepted) >= count:
            break

        prompts_needed = min(prompts_per_call, count - len(accepted))
        if args.mode == "initial":
            prompt_existing = accepted[-5:]
        else:
            prompt_existing = (existing + accepted)[-10:]
        instruction = build_instruction(
            template,
            brand=args.brand,
            batch_size=prompts_needed,
            top_k=top_k,
            parents=parents,
            existing=prompt_existing,
        )
        call_seed = generation_call_seed(
            generator_seed, attempt, generation, args.mode
        )

        if model is None:
            model = Model(
                model_config["path"],
                runtime=generator["runtime"],
                model_config=model_config,
                tokenizer_path=model_config.get("tokenizer"),
            )
        raw = model.generate(
            [instruction],
            sampling_overrides={
                "seed": call_seed,
                "json_schema": generator_json_schema(prompts_needed),
            },
        )[0]
        raw_path = attempts_dir / f"attempt_{attempt:03d}.txt"
        raw_path.write_text(raw, encoding="utf-8")

        parsed = parse_generator_response(raw)
        parsed_records = []
        for source_index, prompt in enumerate(parsed, start=1):
            notes = []
            normalized = dedupe_key(prompt)
            if normalized in seen:
                notes.append("duplicate")
            is_valid = bool(prompt) and not notes
            parsed_records.append(
                {
                    "attempt": attempt,
                    "parsed_index": source_index,
                    "prompt": prompt,
                    "valid": is_valid,
                    "validation_notes": notes,
                }
            )
            if not is_valid:
                continue
            seen.add(normalized)
            global_index = args.candidate_offset + len(accepted)
            candidate_id = (
                f"{slugify(args.brand)}_g{generation:02d}_c{global_index:04d}"
            )
            accepted.append(
                complete_candidate(
                    candidate_id=candidate_id,
                    brand=args.brand,
                    text=prompt,
                    generation=generation,
                    parent_ids=parent_ids,
                    call_seed=call_seed,
                    attempt=attempt,
                    source_index=source_index,
                    generator_model=generator["model"],
                )
            )
            if len(accepted) >= count:
                break

        write_json(
            attempts_dir / f"attempt_{attempt:03d}.json",
            {
                "attempt": attempt,
                "call_seed": call_seed,
                "requested_count": prompts_needed,
                "parsed_count": len(parsed),
                "accepted_so_far": len(accepted),
                "items": parsed_records,
                "raw_completion": raw_path.name,
                "timestamp": utc_now(),
            },
        )
        write_jsonl(accepted_path, accepted)
        print(
            f"[generator] attempt {attempt}: requested={prompts_needed} "
            f"parsed={len(parsed)} accepted_total={len(accepted)}/{count}"
        )

    if not accepted:
        raise SystemExit("prompt generation produced zero valid prompts")
    if len(accepted) < count and not allow_partial:
        raise SystemExit(
            f"population incomplete after {max_attempts} attempts: "
            f"{len(accepted)}/{count}"
        )

    write_jsonl(output_path, accepted)
    write_json(
        complete_path,
        {
            "mode": args.mode,
            "generation": generation,
            "requested": count,
            "accepted": len(accepted),
            "complete_population": len(accepted) == count,
            "allow_partial_population": allow_partial,
            "timestamp": utc_now(),
        },
    )
    print(f"[generator] wrote {len(accepted)} candidates to {output_path}")


if __name__ == "__main__":
    main()
