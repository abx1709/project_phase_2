"""Final Evaluation for a saved QLoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Pretty printing & visualization imports
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-eval-samples", type=int, default=-1)
    
    return parser.parse_args()


def load_image(path: Path, allow_missing: bool):
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        # EXPERIMENT A: Remove 448 cap for evaluation.
        img.thumbnail((2048, 2048), Image.LANCZOS)
        return img
    except (FileNotFoundError, OSError) as exc:
        if allow_missing:
            return Image.new("RGB", (448, 448), color=(0, 0, 0))
        raise FileNotFoundError(f"Cannot load {path}") from exc


def main() -> None:
    args = arguments()
    import torch
    from src.data.vizwiz import build_conversation, prepare_records
    from src.evaluation import vizwiz_ans, compute_all_metrics, infer_answer_type
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    # Load split-aware records
    records = prepare_records(args.dataset, args.cache_dir, args.num_eval_samples, split=args.split)
    
    if args.max_samples > 0:
        records = records[: args.max_samples]

    if not records:
        raise ValueError(f"No records found for split '{args.split}'.")

    model, processor, _ = load_quantized_vlm(
        QLoRASettings(args.model_id, use_bf16=not args.fp16),
        adapter_path=str(args.adapter_path),
        trainable=False
    )
    model.eval()
    device = next(model.parameters()).device
    results = []

    with torch.inference_mode():
        eval_pbar = tqdm(records, desc=f"Evaluating {args.split}")
        for item in eval_pbar:
            image = load_image(args.image_root / item["image"], args.allow_missing_images)
            conv = build_conversation(item["question"], target=None)
            prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            input_length = inputs["input_ids"].shape[-1]
            prediction = processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()
            
            results.append({
                "prediction": prediction,
                "references": item["answers"],
                "ans": vizwiz_ans(prediction, item["answers"]),
                "answer_type": item.get("answer_type", "other")
            })

    # Compute all metrics
    metrics = compute_all_metrics(results)
    elapsed = time.perf_counter() - started
    latency_per_query_ms = (elapsed / len(results)) * 1000 if results else 0
    peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "samples": len(results),
        "mean_vizwiz_ans": metrics["overall_ans"],
        "per_class_ans": metrics["per_class_ans"],
        "type_accuracy": metrics["type_accuracy"],
        "type_macro_f1": metrics["type_macro_f1"],
        "type_macro_precision": metrics["type_macro_precision"],
        "type_macro_recall": metrics["type_macro_recall"],
        "per_class_type_metrics": metrics["per_class_type_metrics"],
        "latency_ms_per_query": round(latency_per_query_ms, 2),
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram_gib": round(peak_vram_gib, 3),
        "predictions": results,
    }

    eval_file = args.output_dir / "final_evaluation.json"
    eval_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Rich Console Summary & Tables ---
    console = Console()
    console.print(f"\n[bold cyan]EVALUATION COMPLETE ({args.split})[/bold cyan]")
    console.print(f"Samples: [yellow]{len(results)}[/yellow]")
    console.print(f"Overall ANS: [green]{metrics['overall_ans']:.4f}[/green]")
    console.print(f"Type Accuracy: [green]{metrics['type_accuracy']:.4f}[/green]")
    console.print(f"Macro F1: [green]{metrics['type_macro_f1']:.4f}[/green]")
    console.print(f"Latency (ms/query): [yellow]{latency_per_query_ms:.2f}[/yellow]")
    console.print(f"Peak VRAM (GiB): [yellow]{peak_vram_gib:.3f}[/yellow]\n")

    # 1. Per-Class Metrics Table
    metrics_table = Table(title="Per-Class Metrics Breakdown", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Answer Type", style="cyan", width=18)
    metrics_table.add_column("ANS Score", justify="right")
    metrics_table.add_column("Precision", justify="right")
    metrics_table.add_column("Recall", justify="right")
    metrics_table.add_column("F1-Score", justify="right")

    per_class_ans = metrics.get("per_class_ans", {})
    per_class_prf = metrics.get("per_class_type_metrics", {})
    all_classes = sorted(list(set(list(per_class_ans.keys()) + list(per_class_prf.keys()))))

    for cls in all_classes:
        ans_score = per_class_ans.get(cls, 0.0)
        prf = per_class_prf.get(cls, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        metrics_table.add_row(
            cls,
            f"{ans_score:.4f}",
            f"{prf.get('precision', 0.0):.4f}",
            f"{prf.get('recall', 0.0):.4f}",
            f"{prf.get('f1', 0.0):.4f}"
        )
    console.print(metrics_table)
    console.print()

    # 2. Confusion Matrix Table
    y_true = [r["answer_type"] for r in results]
    y_pred = [infer_answer_type(r["prediction"]) for r in results]
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_table = Table(title="Confusion Matrix (Rows: True, Cols: Pred)", show_header=True, header_style="bold green")
    cm_table.add_column("True \\ Pred", style="yellow")
    for label in labels:
        cm_table.add_column(label, justify="right")

    for i, label in enumerate(labels):
        row_vals = [str(val) for val in cm[i]]
        cm_table.add_row(label, *row_vals)
    console.print(cm_table)

    # 3. Generate & Save Confusion Matrix Heatmap Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=True
    )
    plt.title(f"Confusion Matrix Heatmap ({args.split} split)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Answer Type", fontsize=11, labelpad=10)
    plt.ylabel("True Answer Type", fontsize=11, labelpad=10)
    plt.tight_layout()

    cm_image_path = args.output_dir / "confusion_matrix.png"
    plt.savefig(cm_image_path, dpi=300)
    plt.close()
    
    console.print(f"\n[bold green]Saved confusion matrix image to:[/bold green] [underline]{cm_image_path.resolve()}[/underline]\n")

if __name__ == "__main__":
    main()