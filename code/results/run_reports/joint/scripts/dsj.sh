#!/bin/bash
cd /home/bob/v2g_safe_rl_paper/code
while pgrep -f jt/jtest.py >/dev/null; do sleep 30; done
for m in deepsets_safe_penrew__train_deepsets_safe_penrew__s0__8acae2c0 deepsets_safe_penrew__train_deepsets_safe_penrew__s1__8acae2c0 attn_safe_penrew__train_attn_safe_penrew__s0__83aaca2e attn_safe_penrew__train_attn_safe_penrew__s1__83aaca2e; do
  /home/bob/v2g_safe_rl_paper/venv/bin/python evaluate.py --checkpoint results/raw/$m/checkpoint_final.pt --lower-bound joint --test-set configs/test_grid.yaml --jobs 4 --out results/raw/${m}_jlb || echo FAIL $m
done
