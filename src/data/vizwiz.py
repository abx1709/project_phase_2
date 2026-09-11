"""Portable VizWiz-Hindi records, multimodal collation with chat template."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

import random

def _cache_key(dataset_path: Path, limit: int, split: str) -> str:
    stat = dataset_path.stat()
    raw = f"{dataset_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{limit}:{split}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def choose_target(record: dict[str, Any]) -> tuple[str, list[str]]:
    hindi = record.get("answers_hi") or []
    source = hindi if hindi else record.get("answers") or []
    answers = [entry.get("answer", "").strip() for entry in source if entry.get("answer", "").strip()]
    if not answers:
        answers = ["unanswerable"]
    return answers[0], answers


def build_conversation(question: str, target: str | None = None):
    instruction = "इस प्रश्न का उत्तर एक शब्द या छोटे वाक्य में दें।"
    full_question = f"{question} {instruction}"
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_question}]}
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages

def prepare_records(dataset_path: Path, cache_dir: Path, limit: int, split: str) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{_cache_key(dataset_path, limit, split)}"
    prepared_path = cache_dir / f"{stem}_prepared.json"
    if prepared_path.exists():
        return json.loads(prepared_path.read_text(encoding="utf-8"))

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    
    # STRICTLY FILTER BY SPLIT FIRST
    if split != "all":
        raw = [item for item in raw if item.get("split") == split]
    
    # SHUFFLE AFTER FILTERING, BEFORE LIMITING
    random.shuffle(raw)
    
    # THEN APPLY LIMIT
    if limit is not None and limit > 0:
        raw = raw[:limit]

    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        target, answers = choose_target(item)
        question = item.get("question_hi") or item.get("question") or ""
        records.append({
            "index": index,
            "image": item["image"],
            "question": question,
            "target": target,
            "answers": answers,
            "answer_type": item.get("answer_type"),
            "split": item.get("split"),
        })
    prepared_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records

class VizWizHindiDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_root: Path, allow_missing_images: bool = False):
        self.records = records
        self.image_root = image_root
        self.allow_missing_images = allow_missing_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        path = self.image_root / item["image"]
        try:
            image = Image.open(path).convert("RGB")
            max_size = 448  # Increased to 448 for better VLM accuracy
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        except (FileNotFoundError, OSError) as exc:
            if not self.allow_missing_images:
                raise FileNotFoundError(
                    f"Cannot load {path}. Supply --image-root containing VizWiz images, "
                    "or use --allow-missing-images only for a wiring smoke test."
                ) from exc
            image = Image.new("RGB", (448, 448), color=(0, 0, 0))
        return {**item, "image_data": image}


class LlavaDataCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for ex in examples:
            conv = build_conversation(ex["question"], ex["target"])
            prompt = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            texts.append(prompt)
            images.append(ex["image_data"])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for i, ex in enumerate(examples):
            user_conv = build_conversation(ex["question"], target=None)
            user_text = self.processor.apply_chat_template(
                user_conv, tokenize=False, add_generation_prompt=True
            )
            user_ids = self.processor.tokenizer(user_text, add_special_tokens=True)["input_ids"]
            user_len = len(user_ids)
            labels[i, :user_len] = -100

        batch["labels"] = labels
        return batch