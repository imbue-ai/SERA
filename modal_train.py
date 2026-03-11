#!/usr/bin/env python3
"""Run SERA axolotl training jobs on Modal with H100 GPUs.

Uploads a local SERA-format JSONL dataset, runs axolotl fine-tuning on
Modal H100s, and converts the checkpoint for vLLM compatibility.

Usage:
    # Train on flask experiment data (2x H100, default settings)
    uv run python modal_train.py \
        --dataset experiments/gemini_flask/data/stage_two_instances_*.jsonl \
        --run-name flask-specialist

    # Custom model / hyperparameters
    uv run python modal_train.py \
        --dataset data.jsonl \
        --model Qwen/Qwen3.5-9B \
        --num-gpus 4 --epochs 2 --lr 5e-6 \
        --run-name custom-run

    # Smoke test: tiny run to validate everything works
    uv run python modal_train.py --dataset data.jsonl --run-name smoke --smoke-test

    # Download trained model (separate script)
    uv run python scripts/download_model.py flask-specialist
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import modal

modal.enable_output()

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("sera-training")

training_data_vol = modal.Volume.from_name("sera-training-data", create_if_missing=True)
models_vol = modal.Volume.from_name("sera-models", create_if_missing=True)

DATA_VOL_PATH = "/data"
MODELS_VOL_PATH = "/models"

axolotl_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "gcc", "g++")
    .pip_install("torch==2.10.0")
    .pip_install(
        "axolotl[deepspeed]",
        "wandb",
        "safetensors",
        "transformers>=4.51",
        "torchvision",
        "pyyaml",
    )
    .run_commands(
        # Fix axolotl telemetry bug: missing whitelist.yaml in pip install
        "python -c \"import axolotl, pathlib; "
        "p = pathlib.Path(axolotl.__file__).parent / 'telemetry' / 'whitelist.yaml'; "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text('organizations: []\\n')\"",
        # Disable axolotl telemetry
        "export AXOLOTL_DO_NOT_TRACK=1 || true",
    )
    .env(
        {
            "NCCL_DEBUG": "WARN",
            "TOKENIZERS_PARALLELISM": "false",
            "AXOLOTL_DO_NOT_TRACK": "1",
        }
    )
)

# ---------------------------------------------------------------------------
# DeepSpeed config (inlined)
# ---------------------------------------------------------------------------

DEEPSPEED_ZERO1 = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
    },
    "bf16": {"enabled": "auto"},
    "fp16": {
        "enabled": "auto",
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# ---------------------------------------------------------------------------
# Axolotl config generation
# ---------------------------------------------------------------------------


def build_axolotl_config(
    *,
    base_model: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int,
    learning_rate: float,
    sequence_len: int,
    gradient_accumulation_steps: int,
    deepspeed_path: str,
    wandb_project: str,
    wandb_name: str,
    smoke_test: bool = False,
) -> dict:
    """Build an axolotl config dict matching SERA's training setup."""
    config = {
        "base_model": base_model,
        "deepspeed": deepspeed_path,
        "load_in_8bit": False,
        "load_in_4bit": False,
        # CutCrossEntropyPlugin may not be available in all axolotl versions
        # "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
        "chat_template": "chatml",
        "datasets": [
            {
                "path": dataset_path,
                "type": "chat_template",
                "field_messages": "messages",
                "ds_type": "json",
                "message_field_training": "train",
            }
        ],
        "dataset_prepared_path": "/tmp/dataset_cache",
        "output_dir": output_dir,
        "sequence_len": sequence_len,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_batch_size": 1,
        "num_epochs": num_epochs,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "learning_rate": learning_rate,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "bf16": "auto",
        "tf32": False,
        "gradient_checkpointing": True,
        "activation_offloading": True,
        "logging_steps": 1,
        "flash_attention": False,  # flash-attn not installed; uses sdpa fallback
        "loss_watchdog_threshold": 5.0,
        "loss_watchdog_patience": 3,
        "warmup_ratio": 0.1875,
        "evals_per_epoch": 0,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "weight_decay": 0.01,
    }

    if wandb_project:
        config["wandb_project"] = wandb_project
        config["wandb_name"] = wandb_name
    else:
        config["wandb_project"] = ""

    if smoke_test:
        config["num_epochs"] = 1
        config["sequence_len"] = 8192  # enough for SWE-agent trajectories
        config["max_steps"] = 5
        config["save_strategy"] = "no"
        config["logging_steps"] = 1
        config["gradient_accumulation_steps"] = 1
        config["excess_length_strategy"] = "drop"

    return config


# ---------------------------------------------------------------------------
# Checkpoint conversion (mirrors convert_axolotl_checkpoint.py)
# ---------------------------------------------------------------------------


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    """Remove _checkpoint_wrapped_module prefix from weight names."""
    import shutil

    from safetensors.torch import load_file, save_file

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def fix_key(key: str) -> str:
        return key.replace("._checkpoint_wrapped_module.", ".")

    safetensor_files = sorted(input_path.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No .safetensors files in {input_dir}, skipping conversion")
        return

    print(f"Converting {len(safetensor_files)} safetensor files...")
    for sf_path in safetensor_files:
        tensors = load_file(sf_path)
        new_tensors = {fix_key(k): v for k, v in tensors.items()}
        save_file(new_tensors, output_path / sf_path.name)

    index_path = input_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        if "weight_map" in index:
            index["weight_map"] = {fix_key(k): v for k, v in index["weight_map"].items()}
        with open(output_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    for file_path in input_path.iterdir():
        if file_path.suffix == ".safetensors" or file_path.name == "model.safetensors.index.json":
            continue
        dest = output_path / file_path.name
        if file_path.is_file():
            shutil.copy2(file_path, dest)

    print(f"Converted checkpoint saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Training function (defined as plain function, registered dynamically)
# ---------------------------------------------------------------------------


def _train_model_impl(
    run_name: str,
    base_model: str,
    num_gpus: int,
    num_epochs: int,
    learning_rate: float,
    sequence_len: int,
    gradient_accumulation_steps: int,
    wandb_project: str,
    smoke_test: bool = False,
):
    """Run axolotl training inside a Modal container."""
    import subprocess

    import yaml

    dataset_path = f"{DATA_VOL_PATH}/{run_name}/training_data.jsonl"
    raw_output_dir = f"{MODELS_VOL_PATH}/{run_name}/raw"
    converted_output_dir = f"{MODELS_VOL_PATH}/{run_name}/converted"

    # Write DeepSpeed config
    ds_path = "/tmp/deepspeed_zero1.json"
    with open(ds_path, "w") as f:
        json.dump(DEEPSPEED_ZERO1, f, indent=2)

    # Build and write axolotl config
    axolotl_config = build_axolotl_config(
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=raw_output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        sequence_len=sequence_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_path=ds_path,
        wandb_project=wandb_project,
        wandb_name=run_name,
        smoke_test=smoke_test,
    )

    config_path = "/tmp/axolotl_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(axolotl_config, f, default_flow_style=False)

    print("=" * 60)
    print(f"SERA Training: {run_name}")
    print(f"  Model: {base_model}")
    print(f"  GPUs: {num_gpus}x H100")
    print(f"  Epochs: {num_epochs}")
    print(f"  LR: {learning_rate}")
    print(f"  Seq len: {sequence_len}")
    print(f"  Smoke test: {smoke_test}")
    print("=" * 60)

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path) as f:
        num_samples = sum(1 for _ in f)
    print(f"Dataset: {num_samples} samples")

    print("\nAxolotl config:")
    with open(config_path) as f:
        print(f.read())

    # Run axolotl training
    if num_gpus > 1:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "--use_deepspeed",
            "-m", "axolotl.cli.train",
            config_path,
        ]
    else:
        cmd = ["python", "-m", "axolotl.cli.train", config_path]

    print(f"\nRunning: {' '.join(cmd)}", flush=True)
    log_path = f"{MODELS_VOL_PATH}/{run_name}/train.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        proc.wait()

    models_vol.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Training failed with return code {proc.returncode}. "
            f"Logs saved to sera-models:/{run_name}/train.log — "
            f"run: uv run modal volume get sera-models {run_name}/train.log ."
        )

    print("\nTraining complete!")

    # Find and convert the final checkpoint
    raw_path = Path(raw_output_dir)
    checkpoints = sorted(raw_path.glob("checkpoint-*")) if raw_path.exists() else []

    if checkpoints:
        final_ckpt = checkpoints[-1]
        print(f"Converting checkpoint: {final_ckpt}")
        convert_checkpoint(str(final_ckpt), converted_output_dir)
    elif raw_path.exists() and list(raw_path.glob("*.safetensors")):
        print(f"Converting model from: {raw_output_dir}")
        convert_checkpoint(raw_output_dir, converted_output_dir)
    else:
        print("Warning: No checkpoint found to convert (smoke test with save_strategy='no'?)")

    models_vol.commit()

    return {
        "run_name": run_name,
        "status": "success",
        "raw_output": raw_output_dir,
        "converted_output": converted_output_dir,
        "num_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def upload_dataset(run_name: str, local_paths: list[str]) -> None:
    """Upload local JSONL file(s) to the Modal training data volume."""
    all_lines = []
    for path in local_paths:
        with open(path) as f:
            all_lines.extend(f.readlines())

    print(f"Uploading {len(all_lines)} samples from {len(local_paths)} file(s)...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.writelines(all_lines)
        tmp_path = tmp.name

    remote_path = f"{run_name}/training_data.jsonl"
    subprocess.run(
        ["uv", "run", "modal", "volume", "put", "sera-training-data", tmp_path, remote_path, "--force"],
        check=True,
    )
    os.unlink(tmp_path)
    print(f"Uploaded to volume sera-training-data:/{remote_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SERA axolotl training on Modal H100s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Train on flask data
              uv run python modal_train.py --dataset experiments/gemini_flask/data/stage_two*.jsonl --run-name flask-v1

              # Smoke test
              uv run python modal_train.py --dataset data.jsonl --run-name smoke --smoke-test

              # Download trained model (separate script)
              uv run python scripts/download_model.py flask-v1
        """),
    )

    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Path(s) to JSONL training data (supports globs)")
    parser.add_argument("--run-name", type=str, required=True, help="Unique name for this training run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B", help="HuggingFace model ID")
    parser.add_argument("--gpu", type=str, default="H200", help="GPU type (H100, H200, A100-80GB)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=32768, help="Max sequence length")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--wandb-project", type=str, default="sera-training", help="W&B project name (empty to disable)")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run (1 epoch, 5 steps, short seq)")

    return parser.parse_args()


def resolve_dataset_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns in dataset paths."""
    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"Error: no files match '{pattern}'")
            sys.exit(1)
        paths.extend(expanded)
    return sorted(set(paths))


def main():
    args = parse_args()

    dataset_paths = resolve_dataset_paths(args.dataset)
    print(f"Resolved {len(dataset_paths)} dataset file(s):")
    for p in dataset_paths:
        print(f"  {p}")

    if args.wandb_project and not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set in .env — W&B logging will be disabled")
        args.wandb_project = ""

    # Upload dataset to Modal volume
    upload_dataset(args.run_name, dataset_paths)

    # Register training function with the right GPU count
    gpu_spec = f"{args.gpu}:{args.num_gpus}" if args.num_gpus > 1 else args.gpu
    secret_dict = {}
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        secret_dict["WANDB_API_KEY"] = wandb_key
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        secret_dict["HF_TOKEN"] = hf_token
        secret_dict["HUGGING_FACE_HUB_TOKEN"] = hf_token
    secrets = [modal.Secret.from_dict(secret_dict)] if secret_dict else []

    train_fn = app.function(
        image=axolotl_image,
        volumes={DATA_VOL_PATH: training_data_vol, MODELS_VOL_PATH: models_vol},
        gpu=gpu_spec,
        timeout=6 * 3600,
        secrets=secrets,
    )(_train_model_impl)

    # Launch training on Modal
    print(f"\nLaunching training on {args.num_gpus}x {args.gpu}...")
    with app.run():
        result = train_fn.remote(
            run_name=args.run_name,
            base_model=args.model,
            num_gpus=args.num_gpus,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            sequence_len=args.seq_len,
            gradient_accumulation_steps=args.grad_accum,
            wandb_project=args.wandb_project,
            smoke_test=args.smoke_test,
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Run name: {result['run_name']}")
    print(f"  Samples trained on: {result['num_samples']}")
    print(f"  Raw checkpoint: sera-models:/{result['raw_output']}")
    print(f"  Converted model: sera-models:/{result['converted_output']}")
    print(f"\nTo download: uv run python scripts/download_model.py {args.run_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
