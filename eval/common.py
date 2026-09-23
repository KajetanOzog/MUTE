import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
TASKS = ("forget", "retain")


def slugify(name):
    value = name.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


class Config:
    def __init__(self, data, path=None):
        self.path = Path(path or DEFAULT_CONFIG).resolve()
        self.judge = data["judge"]
        self.brands = data["brands"]
        self.models = data["models"]
        self.runtime = data["runtime"]
        self.generator = data.get("generator")
        self.search = data.get("search", {})
        self._by_slug = {slugify(name): name for name in self.brands}

    def model_entry(self, name):
        if name not in self.models:
            raise SystemExit(
                f"unknown model {name!r}; configured models: {', '.join(self.models)}"
            )
        return self.models[name]

    def canonical(self, brand_slug):
        if brand_slug not in self._by_slug:
            raise KeyError(f"unknown brand slug: {brand_slug!r}")
        return self._by_slug[brand_slug]

    def brand_cfg(self, brand_slug):
        return self.brands[self.canonical(brand_slug)]


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as file:
        return Config(yaml.safe_load(file), path=path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def shard(records, num_shards, shard_id):
    return [
        record
        for index, record in enumerate(records)
        if index % num_shards == shard_id
    ]


def load_shards(run_dir):
    records = []
    for path in sorted(Path(run_dir).glob("shard_*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def candidate_inputs(records):
    return ([record["prompt"] for record in records],
            [record.get("candidate_prompt") or None for record in records])


def extract_json(text):
    text = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    for candidate in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


class Model:
    def __init__(self, model_path, runtime, model_config, tokenizer_path=None):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.model_path = str(model_path)
        self.model_name = Path(model_path).name
        self.tokenizer_path = str(tokenizer_path or model_path)
        self.chat_template = model_config.get("chat_template", {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        engine_options = {}
        for name in ("language_model_only", "gdn_prefill_backend"):
            if name in runtime:
                engine_options[name] = runtime[name]
        self.llm = LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype=runtime["dtype"],
            tensor_parallel_size=runtime["tensor_parallel_size"],
            gpu_memory_utilization=runtime["gpu_memory_utilization"],
            max_model_len=runtime["max_model_len"],
            max_num_seqs=runtime["max_num_seqs"],
            enforce_eager=runtime["enforce_eager"],
            disable_log_stats=True,
            seed=runtime["seed"],
            **engine_options,
        )
        self.sampling_config = {
            "temperature": runtime["temperature"],
            "max_tokens": runtime["max_tokens"],
            "seed": runtime["seed"],
            "stop": model_config.get("stop"),
        }
        for name in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if name in runtime:
                self.sampling_config[name] = runtime[name]
        self._sampling_params_type = SamplingParams
        self._structured_outputs_type = StructuredOutputsParams

    def format(self, prompt, system=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template,
        )

    def generate(self, prompts, system=None, sampling_overrides=None, *, systems=None):
        if systems is not None and system is not None:
            raise ValueError("Use either system or per-prompt systems, not both")
        if systems is None:
            systems = [system] * len(prompts)
        if len(systems) != len(prompts):
            raise ValueError("systems must have one entry per prompt")
        prompts = [self.format(prompt, instruction)
                   for prompt, instruction in zip(prompts, systems)]
        sampling = dict(self.sampling_config)
        sampling.update(sampling_overrides or {})
        json_schema = sampling.pop("json_schema", None)
        if json_schema is not None:
            sampling["structured_outputs"] = self._structured_outputs_type(
                json=json_schema
            )
        params = self._sampling_params_type(**sampling)
        outputs = self.llm.generate(prompts, params)
        return [output.outputs[0].text.strip() for output in outputs]
