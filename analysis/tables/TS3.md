**TS3.** Literature configuration (EV2Gym V2GProfitMax layout, 25 chargers, workplace profile), for comparison with published EV2Gym results.

| Method | Optimality gap (€ per charger and episode) | J^pen (€/episode) | Profit (€/episode) | Energy not served (kWh) | Overload (kWh) | Inference (ms/step) |
|---|---:|---:|---:|---:|---:|---:|
| SA-Attn (ours) | 0.507 (0.474–0.540) | -49.01 (-49.84 to -48.18) | -47.12 (-47.30 to -46.93) | 0.19 (0.09–0.29) | 0 (0–0) | 2.11 (2.10–2.12) |
| Attention+penalty | 0.706 (0.515–0.898) | -53.99 (-58.78 to -49.21) | -47.83 (-48.75 to -46.92) | 0.01 (0–0.02) | 0.60 (0.02–1.19) | 1.85 (1.78–1.91) |
| Attention+PPO-Lagrangian | 0.790 (0.533–1.047) | -56.08 (-62.50 to -49.65) | -50.32 (-51.03 to -49.62) | 1.8e-03 (0–3.7e-03) | 0.57 (0–1.15) | 1.82 (1.80–1.84) |
| Attention+safety layer without departure bound | 0.363 (0.358–0.368) | -45.42 (-45.55 to -45.29) | -45.42 (-45.54 to -45.29) | 7.7e-04 (6.7e-04–8.7e-04) | 0 (0–0) | 2.13 (2.11–2.15) |
| Deep Sets+safety layer | 0.516 (0.470–0.561) | -49.23 (-50.36 to -48.10) | -46.29 (-47.42 to -45.15) | 0.29 (0.29–0.29) | 0 (0–0) | 1.59 (1.57–1.62) |
| LLF+safety layer | 0.587† | -51.01† | -51.01† | 0† | 0† | 0.60† |
| LLF | 0.587† | -51.01† | -51.01† | 0† | 0† | 0.46† |
| sLLF+safety layer | 0.587† | -51.01† | -51.01† | 0† | 0† | 0.59† |
| sLLF | 0.587† | -51.01† | -51.01† | 0† | 0† | 0.44† |
| Price-threshold+safety layer | 0.507† | -49.02† | -49.02† | 0† | 0† | 0.64† |
| Price-threshold | 195.993† | -4936.17† | 58.80† | 418.76† | 80.74† | 0.47† |
| AFAP+safety layer | 0.586† | -51.00† | -51.00† | 0† | 0† | 0.59† |
| AFAP | 1.054† | -62.68† | -51.03† | 0† | 1.16† | 0.45† |
| ALAP+safety layer | 0.551† | -50.12† | -50.12† | 0† | 0† | 0.59† |
| ALAP | 0.589† | -51.07† | -50.12† | 0† | 0.09† | 0.43† |
| Round Robin+safety layer | 0.586† | -51.00† | -51.00† | 0† | 0† | 0.60† |
| Round Robin | 0.586† | -51.00† | -51.00† | 0† | 0† | 0.44† |
| eMPC-HiGHS+safety layer | 0.456† | -47.73† | -46.61† | 0.11† | 0† | 2.08† |
| eMPC-HiGHS | 104.263† | -2642.91† | -12.25† | 263.07† | 4.0e-04† | 2.00† |

*Values: mean of the seed-averaged policy over all test episodes; a range in parentheses is the min–max over the 2 training seeds; † deterministic method (single run). Ranges for all columns, bootstrap confidence intervals and per-seed values are provided with the result data.*
*Optimality gap: J^pen of the soft-penalty perfect-information optimum minus J^pen of the method, divided by N (€ per charger and episode).*
