# TIMAR Reproduction Guide

## Environment Setup

Please install **uv** by following the official instructions:  
https://docs.astral.sh/uv/getting-started/installation/

After installation, run:

> **Note**  
> If you are outside mainland China, please remove lines 27–29 in `pyproject.toml` (which specify the PyPI mirror) before running the command below to avoid potential network issues.

```bash
uv sync
```

This will automatically create the required environment and install all dependencies.

---

## Model Training

To train the model and reproduce the main results reported in the paper, run:

```bash
torchrun --nproc_per_node=8 --standalone main.py \
  --model timar_large \
  --diffloss_d 3 \
  --diffloss_w 1024 \
  --epochs 400 \
  --warmup_epochs 100 \
  --batch_size 32 \
  --blr 1.0e-4 \
  --diffusion_batch_mul 4 \
  --output_dir experiments/timar_large-diffloss_d=3_w=1024 \
  --use_encoder_only \
  --use_causal_attn
```

---

## Evaluation

To reproduce the main evaluation results:

```bash
torchrun --nproc_per_node=8 --standalone main.py \
  --model timar_large \
  --diffloss_d 3 \
  --diffloss_w 1024 \
  --use_encoder_only \
  --use_causal_attn \
  --test \
  --mask_history_agent \
  --num_iter 1 \
  --num_sampling_steps 5 \
  --resume experiments/timar_large-diffloss_d=3_w=1024 \
  --test_data_path data/test \
  --output_dir experiments/timar_large-diffloss_d=3_w=1024
```

---

## Using the Provided Checkpoint

For precise reproduction of the reported results, we provide a pretrained checkpoint.

First, download the checkpoint:

```bash
hf download coderchen01/TIMAR --local-dir ckpts/
```

Then, run evaluation using the downloaded checkpoint:

```bash
torchrun --nproc_per_node=8 --standalone main.py \
  --model timar_large \
  --diffloss_d 3 \
  --diffloss_w 1024 \
  --use_encoder_only \
  --use_causal_attn \
  --test \
  --mask_history_agent \
  --num_iter 1 \
  --num_sampling_steps 5 \
  --resume ckpts/TIMAR.pth \
  --test_data_path data/test \
  --output_dir experiments/main_results
```