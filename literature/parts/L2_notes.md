# L2 notes - Safe / constrained RL and constraint-enforcing action layers (power & energy focus)

Part: L2 (Literature Agent A1). Date: 2026-10-21. All keys below are verified in `L2_verified.jsonl` / `L2_verified.bib`.
Read level is given per paper: [FT] = full text (via arXiv PDF/HTML through WebFetch), [ABS] = abstract only.
Notes are paraphrases; section/figure pointers refer to the version read (arXiv version when stated).

---

## A. Foundations: CMDP, Lagrangian, safety layer, shielding, barrier functions, base learners

### altman2021constrained [ABS - metadata only]
- Canonical monograph on constrained MDPs (expected discounted/average cost constraints); Crossref record is the 2021 Routledge digital edition (chapter list includes "Convergence of discounted constrained MDPs", "Sensitivity analysis").
- No abstract or full text accessible; cite only as the formal definition of the CMDP setting.
- Note for the writer: the brief calls it "Altman 1999"; the verified DOI record carries year 2021 - cite the edition matching the DOI.

### achiam2017constrained [ABS]
- CPO: trust-region policy search for CMDPs, "near-constraint satisfaction" at each iteration for expected-cost constraints.
- Guarantee rests on a bound relating returns of two policies to their average divergence.
- Evaluated on simulated robot locomotion with safety-motivated constraints.
- For our taxonomy: CMDP / expected-cost class - no per-step feasibility of individual actions.

### tessler2018reward [ABS]
- RCPO: multi-timescale Lagrangian method; the multiplier acts as an adaptive penalty signal.
- Convergence of the scheme is proved (asymptotic); constraint is on expected cost.
- Representative of the Lagrangian / PPO-Lagrangian family used in EV papers (e.g. fan2024safety).

### dalal2018safe [ABS]
- Safety layer appended to the policy that analytically corrects actions per state so that (linearised, learned) constraints are not violated.
- Closed form comes from a linear constraint model learned from logged data (behaviour policy arbitrary).
- Reports zero violations where reward shaping fails (new physics-based environments).
- su2025review summarises the usual caveat that the closed form assumes (at most) one active constraint at a time and relies on linearisation.
- Closest conceptual ancestor of our action layer, but our layer is exact for a known polytope (box + coupling sum) rather than based on a learned linear model.

### alshiekh2018safe [ABS]
- Shield synthesised from a temporal-logic specification; it monitors and overrides only actions that would violate the specification.
- Guarantee holds during learning and execution w.r.t. the safety abstraction used for synthesis; discusses conditions preserving learner convergence.
- Taxonomy: "filter / shield" class (correct-only-when-needed), typically discrete actions.

### wang2024control [ABS]
- Barrier-function family representative: control Lyapunov-barrier function embedded in the performance index; NN approximates the CLBF-based value function.
- Proves stability, safety and optimality results accounting for NN approximation error; chemical process case.
- Different mechanism from ours (shapes cost/value instead of projecting actions). Low relevance; used only to populate the CBF branch of the taxonomy (the more canonical CBF-filter papers could not be read in this pass - see DONE).

### gu2024review [FT of arXiv version 2205.10330]
- Organises safe RL around "2H3W" questions (safe policy, sample complexity, applications, benchmarks, challenges) (Sec. 1).
- Constraint types (Sec. 2.2): cumulative (discounted sum, mean, probabilistic) vs instantaneous (explicit/implicit) vs state-based - useful to position "per-step hard feasibility" vs CMDP expected cost.
- Method families: primal-dual, trust-region (CPO, PCPO, FOCOPS), formal methods / shielding, Lyapunov, GP; model-based vs model-free; single vs multi-agent (Sec. 3-4).
- Notes OptLayer-type safety layers lack convergence analysis and that few such layers come with rigorous training-time guarantees (Sec. 4.3.2, 5.2).
- Benchmarks: AI Safety Gridworlds, Safety Gym, Safe Control Gym, multi-agent safe MuJoCo/Robosuite/Isaac Gym (Sec. 6).

### schulman2017proximal [ABS], haarnoja2018soft [ABS], fujimoto2018addressing [ABS]
- PPO: clipped surrogate, multiple minibatch epochs per batch; on-policy, stochastic.
- SAC: off-policy maximum-entropy actor-critic with stochastic actor; stable across seeds.
- TD3: clipped double-Q (min of two critics), delayed actor updates; deterministic actor.
- None handles constraints; they are the base learners on which penalty, Lagrangian (PPO-/SAC-Lagrangian) and projection layers are built. markgraf2025safe and kasaura2023benchmarking show that whether the projection is differentiated through matters more for deterministic (TD3/DDPG) than stochastic on-policy learners.

## B. Differentiable optimisation / hard-constraint layers

### amos2017optnet [ABS]
- QP as a network layer; exact gradients by implicit differentiation of the KKT system; batched GPU interior-point solver.
- Output feasibility follows from solving the QP exactly (up to solver tolerance); gradient cost small relative to solve.

### agrawal2019differentiable [ABS]
- cvxpylayers: differentiate through any disciplined parametrized convex program via affine-solver-affine form (CVXPY 1.1, PyTorch/TF).
- General but solver-based: chen2021enforcing reports 8.6 ms projection vs 4.5 ms network per forward pass; kasaura2023benchmarking reports large overhead for cvxpylayers-based projection (DOpt).

### donti2021dc3 [ABS]
- DC3: equality constraints via (differentiable) completion, inequality constraints via unrolled gradient-based correction; demonstrated on AC-OPF.
- Feasibility of inequalities depends on correction steps (iterative), contrasted with our closed-form layer.

### li2023learning [ABS]
- Gauge-map approach for hard linear constraints: learn a point in the l-infinity unit ball, gauge-map it onto the (reduced) feasible polytope, recover dependent variables.
- Feasible by construction, single forward pass; not RL. Same gauge idea as tabas2022computationally.

### chen2021enforcing [FT, arXiv 2105.08881]
- PROF: Euclidean projection layer onto an approximate convex constraint set inside the policy (Sec. 3.2, Eq. 7, Eq. 14), implemented with cvxpylayers; gradients via implicit differentiation of the KKT conditions (Eqs. 8-9).
- Adds an auxiliary loss lambda * ||pi - pi_hat||^2 between pre- and post-projection actions (Eq. 16, lambda = 10) to keep gradients informative - an early fix for what later papers call action aliasing / zero gradient.
- Building HVAC (EnergyPlus, PPO with LSTM policy): about 4% less heating energy than Gnu-RL (33,271 vs 34,687 kW); comfort mostly kept but violated when control saturates (Sec. 5.3).
- Inverter Volt/Var on IEEE 37-bus with 21 PV systems: 0 voltage violations vs 22.3% of time for IEEE 1547.8 Volt/Var (Sec. 6.3, Table 1).
- Stochastic policies: both mean and sampled action projected (Sec. 4.1).
- Limitations stated: projection cost per forward pass, dependence on approximate model, no device-count scaling study (Sec. 7). Most direct precursor of contribution 2; our layer differs by being closed-form (no solver), exact for the true box+sum constraint set, and size-agnostic.

### gros2020safe [FT, arXiv 2004.00915]
- Projecting the greedy Q-learning policy onto a safe set is in general suboptimal; fix: minimise Q under the safety constraint and apply the restriction in the TD target (Sec. 3, Lemma 1).
- Deterministic policy gradient through a projection: multiply the unprojected gradient by M(x), the projector onto the null space of active-constraint gradients (Prop. 1, Cor. 1); ignoring it biases the gradient.
- Stochastic policy gradient: use the score of the unprojected sample with the advantage of the projected action (Prop. 2) - i.e. treating the projection as part of the environment is unbiased for stochastic policies.
- Extends to robust-MPC projections (Cor. 2-3); assumes LICQ and strict SOSC; illustrated on one linear example (Sec. 6).
- Directly supports our gradient argument for a PPO-type stochastic policy with a piecewise-linear projection.

### kasaura2023benchmarking [FT, arXiv 2304.08743]
- Benchmarks action-constrained RL: projection with critic on projected (DPro) vs pre-projected actions (DPre), differentiable optimisation layer (DOpt), Frank-Wolfe NFWPO, alpha-projection, radial squashing, plus penalty variants (Sec. III).
- Constraint types include box, L2 ball and weighted linear-sum upper bounds - the same structure as a station power limit (Table II).
- Explicitly discusses the zero-gradient problem: many infeasible actions project to the same point (Sec. III-A).
- Finding: DPre(+penalty) is a strong simple baseline; DOpt is expensive without clear benefit; alpha-projection and radial squashing are cheap and competitive (Sec. IV-E).

### markgraf2025safe [FT, arXiv 2509.12833; preprint]
- Formalises safe-environment RL (projection treated as part of the environment, no gradient through it) vs safe-policy RL (projection as differentiable layer) (Sec. 5).
- Theorem 1: optimal value functions coincide; Lemma 1: SE-RL and SP-RL equivalent for stochastic policies with GAE; Lemma 2: action aliasing (normal-cone actions map to one safe action) gives rank-deficient Jacobians; Lemma 3: flat-lining critic in SE-RL (Sec. 6, App. A).
- Mitigations: SE-RL reward penalty on squared projection distance (Sec. 7.1); SP-RL per-sample loss (after chen2021enforcing) and proposed penalty critic PenC (Sec. 7.2).
- Experiments with TD3 and A2C on pendulum, quadrotor, navigation and an energy-management task: A2C shows identical SE/SP results; with TD3 vanilla SE-RL often wins, improved SP-RL matches or beats penalised SE-RL; SP-RL costs about 3-12x compute (Sec. 8-9).
- Most useful single reference for our discussion of gradient handling; recommends vanilla SE-RL as strong baseline.

### wang2022efficient [ABS]
- Projection onto half-space intersected with a box-like set reduced (Lagrangian duality) to a univariate semismooth equation; semismooth Newton with line search, global and locally quadratic convergence.
- Gives an explicit generalised Jacobian of the projection - the formal object needed to back-propagate through a box+sum projection (relevant to the "true (sub)gradient" option).

### kiwiel2007breakpoint [ABS]
- Linear-time breakpoint-searching algorithms for the continuous quadratic knapsack problem (the problem class of Euclidean projection onto a box intersected with one linear equality constraint).
- Reports cycling / wrong-convergence examples in earlier algorithms - useful caution for implementing the multiplier search for lambda in p_i = clip(a_i - lambda, l_i, u_i).
- Abstract-level only; before quoting the formulation, check the full text (rcin.org.pl copy was behind a bot check).

## C. Safe RL in power and energy systems

### su2025review [FT of arXiv 2407.00304, read through Sec. IV]
- Taxonomy (Sec. III, Fig. 4): penalised reward; learning-process modifications (Lagrangian, Lyapunov, GP, barrier, robust RL); RL-structure adjustments (projection/trust region, shielding, safety layer).
- Sec. III-I / Table II: projection, Lyapunov, shielding and strictly applied safety layers give hard guarantees; Lagrangian, barrier, GP are soft/probabilistic.
- Safety layers: modular and real-time but rely on linearised constraints, per-step optimisation and (often) one-active-constraint assumption (Sec. III-F).
- Scalability of projection/safety layers to large systems flagged as a key barrier (Table II, Sec. IV-A1); our closed-form layer (sort/breakpoint search, our own complexity estimate O(N log N) or O(N)) targets exactly this.
- In the text we read, gradients through projections and differentiable optimisation layers were not discussed.

### tabas2022computationally [FT, arXiv 2110.10333]
- Closed-form, differentiable safety filter for frequency regulation: gauge map from the l-infinity ball onto the state-dependent safe action polytope derived from a robust controlled-invariant set plus a linear backup controller (Sec. III, Eq. 12-13).
- Theorem 1: resulting policy is safe for all states in the RCI set, computable in closed form, differentiable at interior points (subgradients otherwise) - a true "provable per-step feasibility by construction" example in power systems.
- DDPG on a modified 9-bus system; cheaper than robust linear feedback with same safety; penalty-DDPG leaves the safe set (Figs. 5-7).
- Assumes linear dynamics with bounded disturbances; RCI set computed offline by SDP.

### ceusters2023adaptive [FT, arXiv 2304.08897]
- OptLayer QP projection + SafeFallback policy (switch when projection distance exceeds threshold) = OptLayerPolicy; self-improving constraint residuals learned by MLPs (Sec. 3).
- TD3; projected action executed in the environment, original action stored with a penalty (Alg. 1) - an SE-RL style treatment.
- Multi-energy system with wind, PV, boiler, heat pump, CHP, TESS, BESS: initial utility 92.4% vs 86.1% (OptLayer); constraint tolerance 6.1-6.5% vs 22.3% unsafe TD3 (Tables 3-4).
- Empirical only; about 1.9 s per step for the grey-box variant (scalability concern).

### fan2024safety [FT, arXiv 2403.13236]
- EV charging stations in IEEE 33-bus feeder; voltage constraint handled as CMDP cost with SAC-Lagrangian (Sec. III-IV).
- Results: cost 926 $ vs 933 (SAC) / 1091 (DDPG); voltage-violation count 294 vs 436 / 551; unfinished demand about 411 kWh for all (Table I).
- Illustrates the gap between expected-cost satisfaction and per-step feasibility in EV charging: violations reduced but not eliminated; demand not guaranteed.
- Fixed 4 stations x 5 chargers; scalability left to future work.

### guan2024rule [ABS]
- EV charging control as large-scale partially observable CMDP; rule-based local and global shields supervise agents' actions.
- Abstract claims strict security in training and execution and better near-optimality; IEEE 33-node test.
- Shield class applied to EV charging; full text (to check whether departure energy is part of the shield and whether a proof exists) was not accessible.

### salesortiz2024efficient [ABS]
- VPP trading aggregate bidirectional EV flexibility; aggregation makes the RL policy independent of the number of contracts / connected EVs.
- RL action constrained within bounds that ensure individual EV charging requirements, then proportionally fair disaggregation to chargers.
- Closest prior idea to "scale-agnostic + requirement-preserving action bounds", but with an aggregate (single-action) policy rather than per-EV tokens; full text inaccessible, so the formal status of the guarantee is unknown to us.

### yan2026hierarchical [FT of arXiv HTML v1, read through Sec. 5.1]
- HPC-RL for V2G inside OPF: upper level GRG-SAC (basic/non-basic variable split, Newton completion, implicit-function gradient -(J_N)^-1 J_B, projection of violating inequalities) (Sec. 4.1).
- Prop. 4.1 gives only first-order (tangent-space) feasibility; AC feasibility up to Newton tolerance; assumes nonsingular J_N.
- Lower level "dynamic boundary": per-EV bounds P_min = max{0, remaining energy - remaining intervals * P_phys}, P_max = min{P_phys, remaining capacity} (Sec. 4.2, Eq. 16) - the same "latest-start" idea that underlies our l_i.
- Departure-energy guarantee is asserted, but in the text we read there is no theorem for it and the case P_min > P_max is not treated; experiments fix 10 EVs per step.
- Most closely related recent EV/V2G paper to contribution 2; our differences: single closed-form projection that jointly enforces per-EV energy bounds and the station limit with a stated feasibility condition and proof, used identically in training and deployment.

### arif2026constraint [ABS; SSRN preprint]
- Electric-bus depot charging on a weak 11 kV feeder; two-pool CMDP with separate deadlines; small DQN (6,165 parameters).
- Network constraint enforced by a closed-form projection before the plant; zero voltage-floor breaches across test days, seeds and hyperparameters vs 6/day uncoordinated.
- Cost -45.0% vs uncoordinated, 4.4% above perfect-foresight LP bound.
- Key ablation: with the projection active, physics-informed reward shaping becomes redundant (penalty never incurred); without the layer, shaping reduces violations from 1.20/day to 0 at 34 kWh unserved energy.
- Abstract does not state which constraint is projected exactly, whether departure energy is guaranteed, or any proof.

---

## Which papers already apply a (provable) projection layer to EV charging?
- arif2026constraint: closed-form projection for a network constraint in depot EV-bus charging; empirical zero violations; no proof stated (abstract); discrete DQN; preprint.
- yan2026hierarchical: per-EV dynamic bounds (departure energy) + GRG/projection for grid constraints in V2G; guarantee asserted, only first-order feasibility proposition; preprint.
- salesortiz2024efficient: bounds on an aggregate EV action to preserve individual requirements; abstract-level, formal status unknown.
- guan2024rule: rule-based shields for EV charging claiming strict security; abstract-level.
- None of the papers we could read gives a closed-form differentiable projection that provably enforces both per-EV departure energy and a station/transformer limit at every step with stated feasibility assumptions, nor combines it with a size-agnostic per-charger policy. (Excluded-by-brief papers DuMES, HetGPS, SmartGridComm 2023 were not examined here.)

---

## Theme synthesis

Constraint handling in RL for energy systems can be read as a ladder of increasing strength. At the bottom is the penalty approach: constraint violations are added to the reward with a hand-tuned weight. It is simple but gives no guarantee, needs careful tuning, and in power-system studies it still violates limits at test time [tabas2022computationally; fan2024safety; arif2026constraint]. The second rung is the constrained MDP [altman2021constrained], solved by Lagrangian or primal-dual methods [tessler2018reward] or by trust-region updates [achiam2017constrained]. These methods bound the expected cumulative cost, so violations are acceptable as long as they are rare on average and asymptotically controlled. Reviews therefore classify them as soft or probabilistic [gu2024review; su2025review]. The EV case of [fan2024safety] (SAC-Lagrangian: 294 rather than 436 voltage violations, demand still unmet) shows the practical consequence. The third rung is the safety filter or shield. A separate module checks each proposed action and overrides it, using a synthesised shield [alshiekh2018safe; guan2024rule], a learned linear safety layer [dalal2018safe], or a QP-based correction with a fallback [ceusters2023adaptive]. The guarantee is per step, but it holds only with respect to the model the filter uses, and it is often checked only empirically. The strongest rung makes the policy's output feasible by construction through a projection or optimisation layer. Examples are differentiable QP or convex layers [amos2017optnet; agrawal2019differentiable] used inside energy policies [chen2021enforcing], completion and correction schemes [donti2021dc3], and closed-form gauge maps onto polytopes [tabas2022computationally; li2023learning]. Su et al. list projection-type and strictly applied safety layers among the hard-constraint methods, but they flag per-step computation and scalability as the main barriers [su2025review]. Chen et al. report that their solver-based projection takes longer than the policy network itself [chen2021enforcing].

The barrier and Lyapunov families sit beside this ladder. They shape the value function or cost and can give stability and safety proofs under modelling assumptions [wang2024control]. They do not, however, map arbitrary network outputs into a feasible set. For our problem the relevant distinction is among three kinds of guarantee: (i) provable per-step feasibility by construction (a closed-form or exact projection onto a known set, as in [tabas2022computationally] and in our layer); (ii) expected-cost satisfaction (CMDP methods); and (iii) empirical safety (most filters, shields and EV studies, including [arif2026constraint; yan2026hierarchical], whose departure-energy argument is asserted but not proven in the text we read). In EV charging the coupling constraint is a single sum (the station or transformer limit) over per-EV boxes whose lower bound comes from the remaining energy and time. The projection is therefore a continuous quadratic knapsack problem. Linear-time breakpoint algorithms [kiwiel2007breakpoint] and semismooth-Newton methods with an explicit generalised Jacobian [wang2022efficient] solve it exactly, which removes the need for a generic QP solver.

Gradient handling through the projection is discussed explicitly in several papers. Gros et al. show that naive projection biases learning. For deterministic policy gradients the unprojected gradient must be multiplied by the null-space projector of the active constraints. For stochastic policies, the score of the unprojected sample combined with the advantage of the projected action gives an unbiased gradient. The second result means that treating the projection as part of the environment is sound for PPO-type learners [gros2020safe]. Kasaura et al. name the "zero-gradient problem": many infeasible actions map to the same feasible point. They find that training on pre-projected actions with a penalty is a strong, cheap baseline, and that differentiable optimisation layers add large overhead without a clear benefit [kasaura2023benchmarking]. Markgraf et al. formalise the choice between "safeguard the environment" (straight-through-like: no gradient through the projection) and "safeguard the policy" (true Jacobian of the projection). They prove that the two are equivalent for stochastic policies with GAE, show that action aliasing makes the projection Jacobian rank-deficient, and propose penalty-based remedies (a per-sample distance loss after [chen2021enforcing], and a penalty critic) [markgraf2025safe]. Chen et al. had already added a pre/post-projection distance loss for this reason [chen2021enforcing]. Tabas and Zhang use subgradients of their piecewise gauge map [tabas2022computationally]. None of the papers we read uses an explicit straight-through estimator under that name. For a clip(a - lambda, l, u) layer the true Jacobian is cheap and piecewise constant: identity on free coordinates minus a rank-one correction from the coupling constraint, and zero on clipped coordinates. This is our own derivation, not taken from the papers, and must be checked in the method section. The aliasing argument of [markgraf2025safe] and the stochastic-policy result of [gros2020safe] therefore give the justification for whichever option we choose, and for adding a small projection-distance penalty.
