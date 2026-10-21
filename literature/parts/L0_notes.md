# L0 notes: core competitors (EV2Gym family, safe/scalable RL for EV charging)

Read levels: `FT` means full text (via arXiv HTML, a preprint PDF, or a publisher page through WebFetch). `ABS` means abstract only. Section pointers follow the version that was read.

---

## Core papers (9)

### [orfanoudakis2025ev2gym] EV2Gym (IEEE T-ITS 26(2), 2025; online Dec 2024). FT: arXiv 2404.01849 plus EV2Gym 2.0.0 source code
- **Authors.** The paper has five authors: Orfanoudakis, Diaz-Londono, Yılmaz, Palensky and Vergara. The brief listed only "Orfanoudakis et al.". Crossref gives the year as 2025 (vol. 26, no. 2, pp. 2410-2421). OpenAlex gives 2024 (online-first).
- **Time step (Sec. II-A, Table II).** Δt is configurable in minutes. Both case studies use 15-minute steps, with about 85 steps per episode in the paper. The shipped configs use `timescale: 15` and `simulation_length` 96-112.
- **Action (Sec. IV-C; `ev2gym_env.py`, `ev_charger.py`).** The action is one value per EVSE port, in [0,1] without V2G or [-1,1] with V2G. In code, amps = a × max_charge_current (or max discharge current). Currents below the minimum are set to zero, the BMS clips per-EV current, and the charger total current is checked. The action and observation vectors scale with the number of ports, so a standard policy is fixed-size.
- **Transformer model (Sec. II-D, Eq. 7; `transformer.py`).** Each transformer has a max/min power, randomised inflexible loads (Pecan Street), PV (Renewables.ninja) and demand-response events that cut capacity, announced about 15 minutes ahead. The environment does not prevent overloads. It only measures them: ε_ov = Σ max(P − P̄, 0), via `get_how_overloaded`. In the RL reward, overload is penalised with weight 100 (Eq. 27). In the MIP it is a hard constraint (Eq. 20).
- **Departure energy (Sec. II-B, III-B, Eq. 24, 27).** In the MIP, energy at departure ≥ E* is a hard constraint. For RL it is only a soft penalty, −100·exp(−10·ε_usr). Caveat: in the code (v2.0.0, `gurobi_models/profit_max.py`), the hard departure constraint is commented out and replaced by a quadratic satisfaction penalty in the objective, while transformer limits stay hard. Our "perfect-information optimum" therefore needs its own hard departure constraint.
- **Baselines.**
  - Heuristics: AFAP, ALAP, Round Robin, plus in code CAFAP-with-power-limit, RoundRobin_GF and DoNothing/Random.
  - MPC: real-time MPC (eMPC_V2G/G2V and OCMF_V2G/G2V in code; based on [diazlondono2026open]).
  - Oracle: a Gurobi optimum with perfect information (`V2GProfitMaxOracleGB`, `tracking_error.py`, `v2g_grid.py`).
  - RL: SB3 algorithms (A2C, ARS, SAC, DDPG, TD3, TQC, PPO, TRPO).
- **Results.**
  - PST, 20 chargers, 100 runs (Table VI): tracking-energy error is 46±14 kWh for Optimal, 57±16 for Round Robin and 213-295 for RL.
  - V2G, 10 chargers (Table VII): TD3 earns the highest profit but reaches only 38% user satisfaction. Optimal reaches 100% at 33 s solve time.
  - This shows that penalty-based RL in EV2Gym badly violates departure-energy requirements.
- **Code and licence.** github.com/StavrosOrf/EV2Gym and github.com/distributionnetworksTUDelft/EV2Gym (footnote 1). The paper states no licence. The PyPI package `ev2gym` 2.0.0 carries an MIT License ("Copyright (c) 2024 Stavros Orfanoudakis"). Version 2.0.0 adds a 150-charger `V2Ggrid.yaml` config with power flow.

### [orfanoudakis2025scalable] EV-GNN (Communications Engineering 4, 2025). FT: Research Square preprint v1; published abstract
- **Authors.** Five authors: the four in the brief plus Pedro P. Vergara.
- **Setting (Results/Methods).** CPO-level power-setpoint tracking with 25, 100, 500 and 1000 charge points grouped under 3-70 transformers. Episodes have 115 steps of 15 minutes.
- **Method.** A heterogeneous graph with EV, charger, transformer and CPO nodes, built from type-specific MLP encoders and GCN layers. The actor outputs one value per node. The critic uses mean pooling. "Branch pruning" removes subtrees without EVs and assigns them action 0. The method is combined with SAC and TD3, and with TD3 for multi-discrete actions.
- **Constraints.** Actions in [0,1] are scaled to charger current. There is no transformer or departure-energy enforcement, and the reward is only the tracking error. The authors list constraint satisfaction and safe RL as future work.
- **Guarantees.** The abstract's "we prove" is supported only by experiments and ablations. There is no theorem.
- **Scalability.** The architecture can handle variable N, but a separate model was trained for each scale. The generalisation tests change data distributions at 500 charge points, not N. Zero-shot transfer across sizes was not reported in the version read.
- **Results (Table 1, 1000 CPs).** SAC EV-GNN reaches 98.7% user satisfaction and 11 MWh error, versus 95.9% and 17 MWh for SAC. Its actor has about 256k parameters versus about 2.8M for SAC.
- **Other details.** The environment is EV2Gym and Gurobi is used for the MPC/oracle. The preprint promises code on release; no URL is given in the version read.

### [orfanoudakis2026graph] GNN-DT (Energy and AI 23, 2026). FT: arXiv 2502.01778; journal abstract
- **Method (Sec. 4).** Offline Decision Transformer (GPT-2, 3 layers, 4 heads) with GCN state/action embedders. Each EV's action comes from the dot product of the output token with that node's embedding (Eq. 8). Absent EVs are masked. Training trajectories come from random, round-robin and optimal (MIP) policies.
- **Constraints (Sec. 3.2, App. A.1).**
  - Aggregate power-limit overshoot is penalised with weight 100. Departure-energy shortfall is penalised with weight 10.
  - There is no projection.
  - Power violation for GNN-DT is 21.7±22.8, against 10.5 for BaU and 2.0 for Optimal (Table 5). So GNN-DT violates more than the simple heuristic.
- **Zero-shot (Sec. 5.3, Fig. 5b).** Trained on 25 chargers and tested on 5, 50, 75 and 100. Performance degrades with size but stays above BaU. The 250-charger case was retrained (Table 4).
- **Optimality.** The journal abstract says rewards are "within 5 percent of an oracle solver". In the preprint's Table 5, GNN-DT scores −0.027 versus −0.020 for the offline optimum.
- **Other details.** The environment is EV2Gym. Code is at github.com/StavrosOrf/DT4EVs.

### [orfanoudakis2026physics] PI-TD3 (Applied Energy 421, 2026). FT: arXiv 2510.12335v1
- **Setting (Sec. IV).** IEEE 34-bus with 150 V2G chargers, and 123-bus with 500 chargers. Episodes have 300 steps of 15 minutes. EV2Gym is coupled with RL-ADN power flow.
- **Method (Sec. III).** TD3 whose actor is trained by back-propagating through K-step model rollouts of a differentiable power-flow and SoC model.
- **Constraints.**
  - Voltage is handled only as a reward penalty (λ₁ = −5×10⁴).
  - Departure SoC is a soft term (λ₃ = −10).
  - SoC bounds are enforced by clipping in the model.
  - There is no safety layer.
- **Guarantees.** None. Results: 95.6±3.6% user satisfaction versus 99.9% for oracle MPC, and violations per bus of 104.2 versus 98.7 for the oracle (Table II).
- **Scalability.** The state is a flat concatenation, so the architecture is fixed-size. It is evaluated under load multipliers only. Transfer across topologies or fleet sizes is not tested.
- **Code.** github.com/StavrosOrf/EV2Gym_PI-TD3

### [wang2026hetgps] HetGPS (arXiv 2608.00679, 2026). FT
- **Authors.** Five authors: the four in the brief plus Saman Kumara Halgamuge, which confirms the "(+Halgamuge?)" in the brief.
- **Setting (Sec. 4).** Five nested CRE21 feeders with 1-32 transformers and 200-3,218 EVs, 24 hourly steps, and AC power flow in pandapower.
- **Method (Sec. 3, App. C-D).**
  - The policy is a parameter-shared heterogeneous-graph SAC over bus and EV nodes, using edge-aware message passing, a GraphGPS global token and EV cross-attention.
  - It has 383,702 parameters at every scale, versus a centralised SAC actor about 170× larger at K = 32.
  - The safety filter has two parts. A learned graph residual risk model sets the intervention budget. A LinDistFlow projection, run as 10 truncated POCS sweeps, sets the direction.
- **Constraints.**
  - Voltage is handled by the filter plus a PID-Lagrangian cost.
  - Departure energy is handled by reward terms plus a heuristic "service map" that forces charging near departure.
  - The transformer overload coefficient is zero, and thermal/ampacity limits are explicitly not constraints (App. B.2, D.1).
- **Guarantees.** Empirical only. The authors state that the objective does not assert pointwise feasibility (Sec. 3, 6). Proposition 1 gives only the sign of the correction.
  - Bus-step violation rate falls from 3.93-7.74% to 0.52-3.44%.
  - Departure success is 99.06-100%.
- **Zero-shot.** Trained at 8 transformers (1,236 EVs) and transferred to 16 and 32 transformers (2,239 and 3,218 EVs), with 0.57-0.75% violations and at least 99.99% departure success.
- **Optimality.** There is no optimality gap. PI-LP is a linearised perfect-information reference, not an optimum.
- **Code.** No code link in the version read.

### [zhang2025dumes] DuMES (IET Smart Energy Systems 1(3), 2025). ABS only
- **Authors.** Six authors: the four in the brief plus Tomas E. Ward and Long Cheng.
- **Setting and method.** A single charging station with renewables and an ESS. DRL with a decision-level safety layer that detects unsafe actions and replaces them, plus a second safety layer coupled with reward shaping so that raw and safe actions converge. Safety is claimed during both training and deployment.
- **What could not be checked.** The full text was blocked (Wiley), so I could not verify the network size, how the layer is built, whether it is proven, or the violation rates. The abstract gives no numbers. Nothing in the abstract indicates a variable-N policy.

### [wu2023network] Network-constrained CRL (IEEE SmartGridComm 2023). FT
- **Authors.** Five authors: the four in the brief plus Sean Peisert.
- **Setting (Sec. II-III).** Aggregate EV demand at a bus, represented by counts per (remaining charge time, slack) state, co-optimised with DC-OPF. The case study is IEEE 14-bus with one station.
- **Method (Sec. IV).** Constrained TD3 with an augmented-Lagrangian primal-dual update, plus a per-bus projection of predicted EV demand at execution (Eq. 26).
- **Results (Sec. V, Fig. 7).** Empirically 100% feasible, versus 83.87% for unconstrained DRL. Normalised MAE to the perfect-information oracle is about 1.9%.
- **Limitations.** The model is aggregate and has no transformer limit. There is no proof in this paper; the convergence proof is in earlier work. No code is given.

### [christensen2025multiagent] Laxity aggregation (Sustainability 17(9):3847, 2025). FT for Sec. 1-4.6; results section not parsed
- **True title and authors.** "A Multi-Agent, Laxity-Based Aggregation Strategy for Cost-Effective Electric Vehicle Charging and Local Transformer Overload Prevention" by K. Christensen, B. N. Jørgensen and Z. G. Ma.
- **Method.** A rule-based aggregator, with no learning. Under a 400 kVA transformer serving 126 households (all with EVs), it shifts the EVs with the highest laxity out of overload hours (Algorithm 1). It runs in an AnyLogic agent-based model over one year at minute resolution.
- **Results.** The abstract says overloads are fully eliminated and compensation costs under DKK 6000 per year.
- **Guarantee.** The algorithm loops "until resolved or shifting is impossible", so there is no formal guarantee.

### [ferragut2022scheduling] Uncertain departure times (ACM SIGMETRICS PER 49(3), 2022). FT
- **True title and authors.** "Scheduling EV charging with uncertain departure times" by A. Ferragut, L. Narbondo and F. Paganini.
- **Method.** A mean-field (fluid-limit) analysis of EDF and LLF in a facility with C chargers when declared deadlines are wrong (Propositions 1-6).
- **Findings.**
  - In overload, mean attained service does not depend on the uncertainty.
  - Under-reporting the deadline is rewarded, unless policies are curtailed at the declared deadline.
  - Simulations use synthetic data and ACN-Data (C = 30).
- **Relevance.** No RL. This is theoretical support for laxity-based priority, and a caution that our "known departure time" assumption is open to strategic misreporting.

---

## Included EV2Gym-citing works (8; plus EV-GNN, GNN-DT and PI-TD3 above, which also cite EV2Gym)
- **[rossi2025smart] (ABS).** A pre-trained RL scheduler with grid simulation in the loop. It minimises cost and maximises ancillary-service reward while mitigating overloads at chosen nodes. Overload is part of the objective (soft), not a guarantee.
- **[yilmaz2024reinforcement] (ABS).** DDPG for CPO setpoint tracking in a commercial lot. Power constraints are built into the action/state design. Compared only with uncontrolled charging.
- **[guo2025hierarchical] (ABS).** HSCA-MARL for V2G under a single transformer: per-station actor-critics, a global critic that "quantifies system risk", and contribution-based credit. Constraints are learned only through the value/reward.
- **[demirel2025multiobjective] (ABS).** eGridLVGym, with eight SORL algorithms, grid-code shielding and MORL (CAPQL). The best SORL lowers transformer overload from 93% to 75%. This shows that shielding and penalties leave frequent overloads.
- **[manzoor2026physics] (ABS).** A PI-TD3 ablation on IEEE 34-bus with 150 chargers. Removing PV raises cost by 6% and voltage violations by 23%. Disabling V2G raises cost by 12%.
- **[rahali2025comparison] (ABS).** Offline model-based RL (behaviour cloning of MPC, with MPC as policy estimator) versus PPO. Both save about 20%.
- **[diazlondono2026open] (ABS).** Open-source G2V/V2G MPC algorithms that are the origin of the EV2Gym MPC baselines. Constraints are hard inside the MPC horizon.
- **[jing2026hybrid] (ABS).** Not RL. Hybrid-priority MPC with a three-stage power allocation that "strictly prevents" transformer overload for 300 EVs. Departure completion is 99.5% overall (96.5% for urgent EVs). This is a non-learning example of transformer feasibility by construction, without a departure guarantee.

## Extra included works found while testing novelty (EV-GNN citers and OpenAlex keyword search)
- **[yan2026hierarchical] HPC-RL (FT, arXiv 2608.03409).**
  - The upper level is GRG-SAC, which solves AC power balance with Newton steps and projects inequality violations.
  - The lower level computes a "dynamic boundary" per EV: P_min(t) = max{0, (SOC_d − SOC) − d_re·P_phys} and P_max = min{P_phys, SOC_max − SOC} (Eq. 16). It assumes known departure time and target SoC.
  - Demand satisfaction is 100%, and the objective gap to MPC is 7-13%.
  - **Not covered.** There is no proof for the boundary. Proposition 1 gives only first-order feasibility for power balance. The paper does not analyse what happens when the sum of P_min exceeds capacity. There is no transformer limit, no size-agnostic architecture and no zero-shot test. It does not use EV2Gym.
- **[wan2025logicinformed] LIRL (FT, arXiv 2511.00806).**
  - A policy gradient whose latent action is projected (Hungarian algorithm plus an OSQP quadratic program) onto a logic-defined admissible set. Lemma 1 proves feasibility by construction, assuming the feasible set is non-empty.
  - Its EV-station case (details are in a supplement I could not read) reports zero transformer-capacity and current violations, against 24.77% for PDQN and 57.34% for HPPO, with 90.4% per-vehicle success.
  - The policy is fixed-size, and departure energy is not modelled as a constraint.
- **[zhang2025modelfree] (ABS).** PCPO-based safe DRL for G2V with transformer thermal stress, voltage and line limits, and unknown departures. It delivers 92% of energy versus under 80% for benchmarks. PCPO bounds expected costs, so it gives no per-step guarantee.
- **[song2025longterm] MAHEM (ABS, SSRN preprint).** Transformer-based MARL for multi-station charging with attention-based communication and Q-decomposition. Cost is 31.6% lower. The abstract mentions no constraint guarantee.
- **[wang2026twostage] SMPD (abstract plus Sec. 2-3.2).** A supervised EV-charger matching stage (with infeasible pairs masked) followed by SAC power dispatch under station limits. How the SAC stage enforces those limits was not visible in what I read.
- **[baweja2026nexusmarl] NEXUS-MARL (ABS).** A heterogeneous graph transformer for MARL with zero-shot transfer from 200 to 2,000 machines, but in the cloud domain. The EV task is zone-level demand balancing with no physical charging constraints.

---

## Theme synthesis

**A family of fixed-size simulators and penalty-based RL.** The EV2Gym family [orfanoudakis2025ev2gym] has become the de facto open benchmark for RL in EV smart charging. Its default interface uses one action per EVSE port and an observation whose length grows with the number of ports. It also leaves transformer overload and departure-energy shortfall to reward penalties. Its own benchmarks show the cost of that choice. RL agents reach high profit but only 38-77% user satisfaction, while the Gurobi oracle reaches 100%. Most EV2Gym-citing RL work keeps the penalty paradigm: PI-TD3 uses voltage and satisfaction penalties [orfanoudakis2026physics; manzoor2026physics], others build constraints into the action or reward design [yilmaz2024reinforcement; rossi2025smart], and some use learned global critics [guo2025hierarchical]. Shielding without proofs also falls here; in eGridLVGym it still leaves transformer overloads in 75% of cases [demirel2025multiobjective].

**Size-agnostic architectures exist, but without hard constraints.** The same group addressed scalability with graph architectures. EV-GNN [orfanoudakis2025scalable] applies GCN encoders over EV/charger/transformer/CPO graphs to 1000 charge points, but trains a separate model per scale. GNN-DT [orfanoudakis2026graph] adds a Decision Transformer and shows zero-shot transfer from 25 to 100 chargers. However, it handles power limits and departure energy only through penalties and imitation data, and its power-limit violations exceed those of a round-robin heuristic. HetGPS [wang2026hetgps] goes furthest on scale: a parameter-shared heterogeneous-graph SAC with attention, transferred zero-shot to 3,218 EVs. It pairs this with a physics-anchored voltage filter, but the authors explicitly disclaim pointwise feasibility, handle departure energy with a heuristic service map, and do not constrain the transformer. Attention-based MARL for multi-station charging [song2025longterm] and generic heterogeneous graph transformers [baweja2026nexusmarl] reinforce that size-agnostic attention policies are now established. They are not novel on their own.

**Safety layers and projections exist, but for fixed sizes and without proofs that cover both constraints.** Constraint-handling work in this neighbourhood includes:
- Lagrangian or primal-dual methods with execution-time projections [wu2023network].
- Action-replacement safety layers [zhang2025dumes].
- Projection-based policy optimisation on expected costs [zhang2025modelfree].
- Generic feasibility-by-construction projections via per-step quadratic programs [wan2025logicinformed].
- Hierarchical schemes that compute per-EV "must-charge" lower bounds [yan2026hierarchical].

The last of these is the closest analogue to our lower bound l_i. It does not prove the bound, nor does it reconcile the bounds with a shared transformer limit. Non-learning schedulers enforce transformer capacity by construction [jing2026hybrid; christensen2025multiagent], and queueing theory explains why laxity-ordered policies are natural, and fragile when deadlines are misreported [ferragut2022scheduling]. None of these combines a provable per-step guarantee covering both departure energy and transformer capacity with a single policy that transfers across station sizes.

**What this means for our paper.** The defensible gap is the combination, not either part alone. Specifically:
- a closed-form, differentiable projection whose feasibility for both departure energy and the transformer limit is proven under stated assumptions;
- used inside training;
- wrapped around a size-agnostic attention policy;
- evaluated zero-shot up to about 200 chargers in EV2Gym, against a hard-constrained perfect-information optimum.

Related work must cite HetGPS, GNN-DT and EV-GNN for scalability, and HPC-RL and LIRL for feasibility layers, and state the differences precisely. The EV2Gym code's own oracle uses a soft departure penalty, so our benchmark must say that we add a hard departure constraint.
