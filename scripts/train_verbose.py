"""Verbose end-to-end QLoRA/LoRA training with per-epoch validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--num-val-samples", type=int, default=500, help="Samples to eval per epoch for checkpoint selection.")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-vram-gib", type=float, default=40.0)
    parser.add_argument("--multiply-max-train-steps", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n{'=' * 65}\n{title.upper()}\n{'=' * 65}")


def move_to_model_device(batch, model):
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in batch.items()}

def load_image(path: Path, allow_missing: bool):
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((448, 448), Image.LANCZOS)
        return img
    except (FileNotFoundError, OSError) as exc:
        if allow_missing:
            return Image.new("RGB", (448, 448), color=(0, 0, 0))
        raise FileNotFoundError(f"Cannot load {path}") from exc

def evaluate_model(model, processor, records, image_root, allow_missing, device):
    """Evaluate a checkpoint using the same logic as final evaluation."""
    import torch
    from tqdm.auto import tqdm

    from src.data.vizwiz import build_conversation
    from src.evaluation import vizwiz_ans, compute_all_metrics

    model.eval()
    results = []

    eval_pbar = tqdm(
        records,
        desc="Evaluating",
        unit="img",
        dynamic_ncols=True,
    )

    try:
        with torch.inference_mode():
            for item in eval_pbar:
                image = load_image(
                    image_root / item["image"],
                    allow_missing,
                )

                conv = build_conversation(
                    item["question"],
                    target=None,
                )

                prompt = processor.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                ).to(device)

                generated = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )

                input_length = inputs["input_ids"].shape[-1]

                prediction = processor.decode(
                    generated[0][input_length:],
                    skip_special_tokens=True,
                ).strip()

                results.append({
                    "prediction": prediction,
                    "references": item["answers"],
                    "ans": vizwiz_ans(
                        prediction,
                        item["answers"],
                    ),
                    "answer_type": item.get(
                        "answer_type",
                        "other",
                    ),
                })

                mean_ans = sum(
                    r["ans"] for r in results
                ) / len(results)

                eval_pbar.set_postfix(
                    ans=f"{mean_ans:.4f}"
                )

    finally:
        eval_pbar.close()
        model.train()

    metrics = compute_all_metrics(results)

    # return metrics["overall_ans"]
    return metrics

def main() -> None:
    args = arguments()
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.data.vizwiz import LlavaDataCollator, VizWizHindiDataset, prepare_records
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.cuda.reset_peak_memory_stats()

    print_section("Data Preparation")
    train_records = prepare_records(args.dataset, args.cache_dir, args.max_train_samples, "train")
    val_records = prepare_records(args.dataset, args.cache_dir, args.num_val_samples, "val")
    
    if not train_records:
        raise ValueError("No training records found.")

    print_section("Loading VLM")
    model, processor, compute_dtype = load_quantized_vlm(
        QLoRASettings(model_id=args.model_id, use_4bit=args.use_4bit, use_bf16=not args.fp16),
        trainable=True
    )

    train_set = VizWizHindiDataset(train_records, args.image_root, args.allow_missing_images)
    loader = DataLoader(
        train_set,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=LlavaDataCollator(processor),
    )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    if args.max_train_steps <= 0:
        args.max_train_steps = (len(train_records) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) + 1
        if args.multiply_max_train_steps:
            args.max_train_steps *= args.num_train_epochs

    print_section("Training Loop Started")
    model.train()
    best_val_ans = -1.0
    step = 0
    losses = []
    started = time.perf_counter()

    for epoch in range(args.num_train_epochs):
        print(f"--- Epoch {epoch + 1}/{args.num_train_epochs} ---")
        train_loader = tqdm(loader, desc=f"Training Epoch {epoch + 1}")
        for batch_index, batch in enumerate(train_loader):
            batch = move_to_model_device(batch, model)
            with torch.autocast("cuda", dtype=compute_dtype):
                loss = model(**batch).loss / args.gradient_accumulation_steps

            loss.backward()

            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                step_loss = loss.item() * args.gradient_accumulation_steps
                losses.append(step_loss)
                # print(f"[STEP {step}/{args.max_train_steps}] Loss: {step_loss:.4f}")
                
                if step >= args.max_train_steps:
                    break
                    
        # --- VALIDATION & CHECKPOINTING ---
        print("Running validation...")
        all_val_ans = evaluate_model(model, processor, val_records, args.image_root, args.allow_missing_images, next(model.parameters()).device)
        val_ans = all_val_ans["overall_ans"]
        
        print(f"Epoch {epoch+1} Validation")
        
        def print_nested(data, indent=1):
            for k, v in data.items():
                formatted_k = k.replace("_", " ").title()
                prefix = "\t" * indent
                if isinstance(v, dict):
                    print(f"{prefix}{formatted_k}:")
                    print_nested(v, indent + 1)
                else:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"{prefix}{formatted_k}: {val_str}")
            if indent == 1:
                print()
                
        def print_non_nested(data):
            for k, v in data.items():
                formatted_k = k.replace("_", " ").title()
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"{formatted_k}: {val_str}")
            print()        

        print_non_nested(all_val_ans)
        # print_nested(all_val_ans)
                
        if val_ans > best_val_ans:
            best_val_ans = val_ans
            print(f"New best score! Saving adapter to {args.output_dir}")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

        if step >= args.max_train_steps:
            break

    peak_vram_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    metrics = {
        "mode": "full_train_verbose",
        "model_id": args.model_id,
        "train_samples": len(train_records),
        "best_val_ans": best_val_ans,
        "epochs_completed": epoch + 1,
        "optimizer_steps": step,
        "losses": losses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": peak_vram_gib,
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_section("Training Complete")
    print(f"Best Validation ANS: {best_val_ans:.4f}")
    print(f"Peak VRAM: {peak_vram_gib} GiB")

if __name__ == "__main__":
    main()