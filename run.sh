#!/usr/bin/env bash
# Reproduces the main Amazon -> Yelp experiment from the paper
# (budget 1000, 30% target, RoBERTa-base, 3 epochs, lr 2e-5, all-MiniLM-L6-v2 embeddings).
set -euo pipefail

python main.py \
  --experiment_name Experimental_Results \
  --dataset amazon-yelp \
  --budget 1000 \
  --target_ratio 0.3 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --model all-MiniLM-L6-v2 \
  --source_method fl1mi \
  --source_eta 1.0 \
  "$@"
