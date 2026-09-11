"""Verbose end-to-end QLoRA/LoRA training with per-epoch validation and cache-dir crash-resume.

Resume state lives entirely in <cache-dir>/train_resume/ so output-dir structure is untouched.
Stop the run at any point (notebook interrupt, kernel restart, OOM kill) and re-run the same
command: it resumes from the last periodic save (every --resume-every optimizer steps, default 100).
If the cached state was made under a different config (e.g. smoke vs full sample counts), it is
detected and discarded automatically so a fresh run starts cleanly.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from datetime import date, datetime

import numpy as np
import torch
from tqdm import tqdm

from huggingface_hub import HfApi, upload_folder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
RESUME_DIR_NAME = "train_resume"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default= "/marimo/cache/train")
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
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Hugging Face Hub repo to push to (must be writable).")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub access token (or set HF_TOKEN env var).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for per-epoch shuffle (must stay fixed across resumes).")
    parser.add_argument("--resume-every", type=int, default=13, help="Optimizer steps between resume saves (kill loses at most this many).")
    parser.add_argument("--push-to-hf-every", type=int, default=0, help="Optimizer steps between pushing to Hugging Face Hub (0 disables).")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached resume state and start over.")
    return parser.parse_args()

def push_to_huggingface(
    repo_id: str,
    token: str,
    folder_path: str = "/root/cache",
    repo_type: str = "model",
):
    """
    Upload a local folder to a Hugging Face Hub repository.

    Args:
        repo_id: Hugging Face repo, e.g. "username/my-model"
        token: Hugging Face access token
        folder_path: Local folder to upload
        repo_type: "model", "dataset", or "space"
    """
    
    try:

        api = HfApi(token=token)

        # Create the repository if it doesn't already exist
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
        )

        # Upload the folder
        upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type=repo_type,
            token=token,
        )

        return f"https://huggingface.co/{repo_id}"
    
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        return None


def print_section(title: str) -> None:
    tqdm.write(f"\n{'=' * 65}\n{title.upper()}\n{'=' * 65}")


def move_to_model_device(batch, model):
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in batch.items()}


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


def evaluate_model(model, processor, records, image_root, allow_missing, device):
    """Evaluate a checkpoint using the same logic as final evaluation."""
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


# ------------------------- resume helpers (cache-dir based) -------------------------

def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


def save_resume(model, optimizer, resume_dir: Path, state: dict) -> None:
    """Atomic save: write to a .tmp dir then swap, so a kill mid-save can never corrupt the last good state."""
    tmp = resume_dir.parent / (resume_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    model.save_pretrained(tmp)
    torch.save(
        {"optimizer_state": optimizer.state_dict(), **state},
        tmp / "trainer_state.pt",
    )

    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    tmp.rename(resume_dir)
    tqdm.write(f"Resume state saved (epoch {state['epoch'] + 1}, batch {state['batches_done']}, step {state['step']})")


# ----------------------------------- main -----------------------------------

def main() -> None:
    args = arguments()
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.data.vizwiz import LlavaDataCollator, VizWizHindiDataset, prepare_records
    from src.models.qlora_vlm import QLoRASettings, load_quantized_vlm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.cuda.reset_peak_memory_stats()
    
    tqdm.write(f"Cache dir: {args.cache_dir}\n")

    print_section("Data Preparation")
    train_records = prepare_records(args.dataset, args.cache_dir, args.max_train_samples, "train")
    tqdm.write(f"Loaded {len(train_records)} training records.")
    val_records = prepare_records(args.dataset, args.cache_dir, args.num_val_samples, "val")
    tqdm.write(f"Loaded {len(val_records)} validation records.")

    if not train_records:
        raise ValueError("No training records found.")

    print_section("Loading VLM")
    model, processor, compute_dtype = load_quantized_vlm(
        QLoRASettings(model_id=args.model_id, use_4bit=args.use_4bit, use_bf16=not args.fp16),
        trainable=True
    )
    tqdm.write(f"Model loaded on device: {next(model.parameters()).device}")

    train_set = VizWizHindiDataset(train_records, args.image_root, args.allow_missing_images)
    tqdm.write(f"Training dataset size: {len(train_set)} samples.")
    loader = DataLoader(
        train_set,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=LlavaDataCollator(processor),
    )
    tqdm.write(f"Training DataLoader created with batch size {args.per_device_train_batch_size}.")

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    tqdm.write(f"Optimizer initialized with learning rate {args.learning_rate}.")

    if args.max_train_steps <= 0:
        args.max_train_steps = (len(train_records) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) + 1
    if args.multiply_max_train_steps:
        args.max_train_steps *= args.num_train_epochs

    tqdm.write("")

    # ------------------------- resume state (lives in cache dir) -------------------------
    resume_dir = args.cache_dir / RESUME_DIR_NAME
    stale_tmp = resume_dir.parent / (RESUME_DIR_NAME + ".tmp")
    if stale_tmp.exists():  # leftover from a kill mid-save — never trusted
        shutil.rmtree(stale_tmp)

    # Config fingerprint: if this changes (smoke vs full, batch size, accum, seed), the cached
    # position is meaningless, so it is discarded and training starts fresh.
    signature = {
        "model_id": args.model_id,
        "train_samples": len(train_records),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,  # optimizer state would silently override lr anyway
        "seed": args.seed,
    }

    step = 0
    start_epoch = 0
    batches_done = 0
    best_val_ans = -1.0
    losses: list[float] = []
    epochs_completed = 0

    state_file = resume_dir / "trainer_state.pt"
    if args.fresh:
        if resume_dir.exists():
            shutil.rmtree(resume_dir)
            tqdm.write("--fresh: cleared cached resume state, starting over.")
    elif state_file.exists():
        payload = torch.load(state_file, map_location="cpu", weights_only=False)  # our own file, contains rng tuples
        if payload.get("signature") != signature:
            tqdm.write(
                "Cached resume state is from a different config "
                f"(had {payload.get('signature')}, now {signature}).\n"
                "Discarding it and starting fresh."
            )
            shutil.rmtree(resume_dir)
        else:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(resume_dir / "adapter_model.safetensors"), strict=False)
            optimizer.load_state_dict(payload["optimizer_state"])
            step = payload["step"]
            start_epoch = payload["epoch"]
            batches_done = payload["batches_done"]
            best_val_ans = payload["best_val_ans"]
            losses = payload["losses"]
            restore_rng(payload["rng"])
            epochs_completed = start_epoch
            tqdm.write(
                f"RESUMED: epoch {start_epoch + 1}/{args.num_train_epochs} (batch {batches_done}), "
                f"step {step}, best_val_ans {best_val_ans:.4f}"
            )
    else:
        tqdm.write("No cached resume state found — starting fresh.")

    def resume_state(epoch_value: int, batches_value: int) -> dict:
        return {
            "epoch": epoch_value,
            "batches_done": batches_value,
            "step": step,
            "best_val_ans": best_val_ans,
            "losses": losses,
            "rng": capture_rng(),
            "signature": signature,
        }
    # ------------------------- end resume state setup -------------------------

    print_section("Training Loop Started")
    model.train()
    epoch = start_epoch  # defined even if interrupted before the first epoch starts
    started = time.perf_counter()

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            tqdm.write(f"--- Epoch {epoch + 1}/{args.num_train_epochs} ---")

            # Seeded per-epoch shuffle: identical data order on a mid-epoch resume.
            gen = torch.Generator()
            gen.manual_seed(args.seed + epoch)
            epoch_loader = DataLoader(
                train_set,
                batch_size=args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=LlavaDataCollator(processor),
                generator=gen,
            )

            batch_iter = iter(epoch_loader)
            for _ in range(batches_done):  # fast-forward past already-consumed batches (CPU/image decode only)
                next(batch_iter)

            train_loader = tqdm(
                batch_iter,
                total=len(epoch_loader),
                initial=batches_done,
                desc=f"Training Epoch {epoch + 1}",
                dynamic_ncols=True,
            )

            for batch_index, batch in enumerate(train_loader, start=batches_done):
                batches_done = batch_index + 1
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

                    # Periodic resume save: any kill loses at most --resume-every steps.
                    if step % args.resume_every == 0:
                        save_resume(model, optimizer, resume_dir, resume_state(epoch, batches_done))
                        
                    args.push_to_hf_every = min(args.push_to_hf_every, args.resume_every)  # don't push more often than we save
                        
                    if args.push_to_hf_every > 0 and step % args.push_to_hf_every == 0:
                        url = push_to_huggingface(repo_id=args.hf_repo_id, token=args.hf_token)
                        # tqdm.ascii.write(f"Checkpoint pushed to Hugging Face Hub: {url}")
                        with open(args.output_dir / f"last_push_urls / last_push_url_{datetime.now()}.txt", "w", encoding="utf-8") as f:
                            f.write(url)

                    if step >= args.max_train_steps:
                        break

            # --- VALIDATION & CHECKPOINTING ---
            tqdm.write("Running validation...")
            all_val_ans = evaluate_model(model, processor, val_records, args.image_root, args.allow_missing_images, next(model.parameters()).device)
            val_ans = all_val_ans["overall_ans"]

            tqdm.write(f"Epoch {epoch+1} Validation")

            def print_nested(data, indent=1):
                for k, v in data.items():
                    formatted_k = k.replace("_", " ").title()
                    prefix = "\t" * indent
                    if isinstance(v, dict):
                        tqdm.write(f"{prefix}{formatted_k}:")
                        print_nested(v, indent + 1)
                    else:
                        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                        tqdm.write(f"{prefix}{formatted_k}: {val_str}")
                if indent == 1:
                    tqdm.write("")

            def print_non_nested(data):
                for k, v in data.items():
                    if isinstance(v, dict):
                        continue  # Skip nested dictionaries for non-nested printing
                    formatted_k = k.replace("_", " ").title()
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    tqdm.write(f"{formatted_k}: {val_str}")
                tqdm.write("")

            print_non_nested(all_val_ans)
            # print_nested(all_val_ans)

            if val_ans > best_val_ans:
                best_val_ans = val_ans
                tqdm.write(f"New best score! Saving adapter to {args.output_dir}")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.output_dir)
                processor.save_pretrained(args.output_dir)

            # Clean epoch boundary: save "next epoch, batch 0" so resume doesn't re-run this epoch.
            epochs_completed += 1
            save_resume(model, optimizer, resume_dir, resume_state(epoch + 1, 0))
            batches_done = 0

            if step >= args.max_train_steps:
                break

    except KeyboardInterrupt:
        tqdm.write("\nInterrupted — best-effort resume save before exiting...")
        try:
            save_resume(model, optimizer, resume_dir, resume_state(epoch, batches_done))
            tqdm.write("Resume state saved. Re-run the same command to continue.")
        except Exception as exc:
            # The last periodic save (<= --resume-every steps ago) is still intact and will be used.
            tqdm.write(f"Save on interrupt failed ({exc}); falling back to last periodic save.")
        sys.exit(130)

    peak_vram_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    metrics = {
        "mode": "full_train_verbose",
        "model_id": args.model_id,
        "train_samples": len(train_records),
        "best_val_ans": best_val_ans,
        "epochs_completed": epochs_completed,
        "optimizer_steps": step,
        "losses": losses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": peak_vram_gib,
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_section("Training Complete")
    tqdm.write(f"Best Validation ANS: {best_val_ans:.4f}")
    tqdm.write(f"Peak VRAM: {peak_vram_gib} GiB")


if __name__ == "__main__":
    main()