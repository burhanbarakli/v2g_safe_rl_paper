#!/bin/bash
cd /home/bob/v2g_safe_rl_paper/code
while systemctl --user is-active --quiet v2g-dsj; do sleep 60; done
for d in results/raw/attn_gin_safe_penrew__*__s[01]__* results/raw/attn_gctx_safe_penrew__*__s[01]__*; do
  m=$(basename $d)
  /home/bob/v2g_safe_rl_paper/venv/bin/python evaluate.py --checkpoint $d/checkpoint_final.pt --lower-bound joint --test-set configs/test_grid.yaml --jobs 4 --out results/raw/${m}_jlb || echo FAIL $m
done
