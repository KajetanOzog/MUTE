import hashlib
from pathlib import Path

from common import DEFAULT_DATASET_ROOT, read_jsonl, slugify, portable_path


def _stable_id(split, source, row_number, task, brand_slug, prompt):
    digest = hashlib.sha1(
        f"{source}:{row_number}:{prompt}".encode("utf-8")
    ).hexdigest()[:10]
    return "__".join(
        part for part in (split, brand_slug or "all", task, digest) if part
    )


def _require_question_answer(record, path, row_number):
    question = record.get("question")
    answer = record.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{path}:{row_number}: missing non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{path}:{row_number}: missing non-empty answer")
    return question.strip(), answer.strip()


def _adapt_train_file(path, task, brand_slug=None, category=None):
    path = Path(path)
    records = []
    for row_number, source_record in enumerate(read_jsonl(path), start=1):
        question, answer = _require_question_answer(
            source_record, path, row_number
        )
        records.append(
            {
                "id": _stable_id(
                    "train",
                    portable_path(path),
                    row_number,
                    task,
                    brand_slug,
                    question,
                ),
                "brand_category": category,
                "brand": brand_slug,
                "task": task,
                "prompt": question,
                "response": "",
                "reference": [answer],
                "source_split": "train",
                "source_file": portable_path(path),
                "source_row": row_number,
            }
        )
    return records


def discover_train_forget(dataset_root, config, brand):
    canonical = config.canonical(slugify(brand))
    brand_slug = slugify(canonical)
    forget_root = Path(dataset_root) / "train" / "forget"
    matches = []
    for path in sorted(forget_root.glob("**/*.jsonl")):
        file_slug = slugify(path.stem.removeprefix("forget_"))
        if file_slug == brand_slug:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"no TRAIN forget dataset found for {canonical!r} under {forget_root}"
        )
    if len(matches) > 1:
        joined = ", ".join(map(str, matches))
        raise ValueError(
            f"multiple TRAIN forget datasets found for {canonical!r}: {joined}"
        )
    return canonical, matches[0]


def load_train_records(
    config,
    brand,
    dataset_root=DEFAULT_DATASET_ROOT,
    forget_dataset=None,
    retain_dataset=None,
):
    canonical = config.canonical(slugify(brand))
    brand_slug = slugify(canonical)
    category = config.brands[canonical]["category"]
    if forget_dataset:
        forget_path = Path(forget_dataset)
    else:
        _, forget_path = discover_train_forget(dataset_root, config, brand)

    if retain_dataset:
        retain_path = Path(retain_dataset)
    else:
        configured = config.search.get("datasets", {}).get("train_retain")
        if not configured:
            raise ValueError(
                "search.datasets.train_retain is not configured and no "
                "--retain-dataset was provided"
            )
        retain_path = Path(configured)
        if not retain_path.is_absolute():
            retain_path = config.path.parent / retain_path

    if not forget_path.is_file():
        raise FileNotFoundError(f"TRAIN forget dataset does not exist: {forget_path}")
    if not retain_path.is_file():
        raise FileNotFoundError(f"TRAIN retain dataset does not exist: {retain_path}")

    forget = _adapt_train_file(
        forget_path, "forget", brand_slug=brand_slug, category=category
    )
    retain = _adapt_train_file(retain_path, "retain")
    return forget + retain

