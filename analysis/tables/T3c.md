**T3c.** Event counts per method over all evaluated episodes (all test configurations, both training seeds).

|  | Method | Episodes | Admitted EVs | Infeasible steps | Fallback activations | EVs with an infeasible step | Unserved EVs | Overload steps | Unserved EVs in episodes without infea- sible step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with safety layer | SA-Attn (ours) | 7,100 | 233,016 | 17,678 | 17,678 | 39,771 | 6,158 | 0 | 0 |
| with safety layer | Deep Sets+safety layer | 7,100 | 233,016 | 12,122 | 12,122 | 27,949 | 5,038 | 0 | 0 |
| with safety layer | MLP+safety layer | 4,200 | 34,954 | 6,767 | 6,767 | 3,589 | 629 | 0 | 0 |
| with safety layer | SA-Attn trained at N = 50 (reference) | 2,700 | 194,466 | 2,689 | 2,689 | 22,583 | 3,643 | 0 | 0 |
| with safety layer | LLF+safety layer | 3,550 | 116,508 | 1 | 1 | 2 | 0 | 0 | 0 |
| with safety layer | sLLF+safety layer | 3,550 | 116,508 | 1 | 1 | 2 | 0 | 0 | 0 |
| with safety layer | Price-threshold+safety layer | 3,550 | 116,508 | 1,675 | 1,675 | 3,025 | 368 | 0 | 0 |
| with safety layer | AFAP+safety layer | 3,550 | 116,508 | 2 | 2 | 7 | 0 | 0 | 0 |
| with safety layer | ALAP+safety layer | 3,550 | 116,508 | 936 | 936 | 1,086 | 84 | 0 | 0 |
| with safety layer | Round Robin+safety layer | 3,550 | 116,508 | 1 | 1 | 2 | 0 | 0 | 0 |
| with safety layer | eMPC-HiGHS+safety layer | 3,550 | 116,508 | 8,595 | 8,595 | 18,765 | 2,526 | 0 | 0 |
| with safety layer | Attention+penalty, safety layer at deployment | 7,100 | 233,016 | 4,020 | 4,020 | 3,606 | 516 | 0 | 0 |
| with safety layer | SA-Attn, no admission control | 7,100 | 233,016 | 17,531 | 17,531 | 40,271 | 5,496 | 0 | 0 |
| safety layer without departure bound | Attention+safety layer without departure bound | 7,100 | 233,016 | 0 | 0 | 0 | 1,551 | 0 | 1,551 |
| conformal departure bound | SA-Attn-CP (α = 0.01) | 7,100 | 233,016 | 9,628 | 9,628 | 27,978 | 84,082 | 0 | 81,742 |
| conformal departure bound | SA-Attn-CP (α = 0.05) | 7,100 | 233,016 | 14,928 | 14,928 | 38,677 | 6,509 | 0 | 4,916 |
| conformal departure bound | SA-Attn-CP (α = 0.1) | 7,100 | 233,016 | 17,333 | 17,333 | 43,695 | 7,887 | 0 | 5,491 |
| conformal departure bound | SA-Attn-CP (α = 0.2) | 7,100 | 233,016 | 21,259 | 21,259 | 51,105 | 12,956 | 0 | 8,126 |
| conformal departure bound | SA-Attn-point (uncalibrated) | 7,100 | 233,016 | 17,339 | 17,339 | 43,936 | 7,877 | 0 | 5,469 |
| without safety layer | Attention+penalty | 7,100 | 233,016 | 3,382 | 0 | 2,818 | 1,451 | 3,730 | 966 |
| without safety layer | Attention+PPO-Lagrangian | 7,100 | 233,016 | 0 | 0 | 0 | 1,013 | 3,274 | 1,013 |
| without safety layer | MLP+penalty | 4,200 | 34,954 | 100 | 0 | 109 | 535 | 3,648 | 478 |
| without safety layer | LLF | 3,550 | 116,508 | 0 | 0 | 0 | 0 | 0 | 0 |
| without safety layer | sLLF | 3,550 | 116,508 | 1 | 0 | 2 | 0 | 0 | 0 |
| without safety layer | Price-threshold | 3,550 | 116,508 | 13,228 | 0 | 33,738 | 97,401 | 7,995 | 58,852 |
| without safety layer | AFAP | 3,550 | 116,508 | 0 | 0 | 0 | 0 | 2,653 | 0 |
| without safety layer | ALAP | 3,550 | 116,508 | 598 | 0 | 874 | 0 | 1,189 | 0 |
| without safety layer | Round Robin | 3,550 | 116,508 | 0 | 0 | 0 | 13 | 0 | 13 |
| without safety layer | eMPC-HiGHS | 3,550 | 116,508 | 28,719 | 0 | 48,416 | 116,107 | 16,224 | 60,858 |
| without safety layer | Attention+penalty, no admission control | 7,100 | 233,016 | 3,560 | 0 | 2,943 | 1,019 | 3,613 | 603 |

*Sums over episodes. Unserved EVs: against the shared admitted requirement. Last column: episode-level check of Theorem 1(b); it is 0 whenever every unserved EV lies in an episode with an infeasible step. Theorem 1(b) covers the methods with the safety layer and the departure lower bound, not the conformal variants or the methods without the bound or the layer.*
