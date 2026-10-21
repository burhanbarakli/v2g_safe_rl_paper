# L3 notes: scalable, permutation-invariant and attention-based policies, and size generalisation

Read levels: full_text = whole paper read through WebFetch (arXiv/MDPI HTML or PDF). abstract_only = abstract read (arXiv abstract page or Semantic Scholar API abstract field). Section, table and figure pointers refer to the version read.

## A. Foundations

### [vaswani2017attention] Attention Is All You Need (abstract_only; https://arxiv.org/abs/1706.03762)
- Proposes the Transformer, an encoder-decoder made only of (multi-head) attention, with no recurrence or convolution.
- The authors report better translation quality on WMT 2014 En-De and En-Fr, with more parallel and cheaper training.
- Relevance: self-attention over a set of tokens is the building block of our charger-as-token encoder. Without positional encodings, self-attention is permutation-equivariant. That is our design choice, not a claim made in this abstract.

### [zaheer2017deep] Deep Sets (abstract_only; S2 API abstract)
- Studies functions defined on finite sets that must not depend on element order.
- Its main theorem characterises permutation-invariant functions and gives the family they must belong to. That family motivates an architecture that sums per-element embeddings and then post-processes the sum.
- Applied to population statistics, point clouds, set expansion and image tagging.
- Relevance: the theoretical basis for a policy that works for any number of chargers. A sum or mean pooled critic is the simplest alternative to attention.

### [lee2019set] Set Transformer (abstract_only; S2 API abstract)
- An attention-based encoder-decoder that models interactions between the elements of a set and stays permutation-invariant.
- Inducing-point attention cuts self-attention cost from quadratic to linear in set size.
- Relevance: the closest generic architecture to our charger-token encoder, and the scaling route if N grows to many hundreds of chargers.

### [kool2019attention] Attention, Learn to Solve Routing Problems! (abstract_only; https://arxiv.org/abs/1803.08475)
- An attention-based model trained with REINFORCE, using a greedy-rollout baseline. The authors report this beats a learned value baseline.
- Near-optimal on TSP up to 100 nodes. One set of hyper-parameters works for VRP variants, OP and stochastic PCTSP.
- The abstract makes no claim of generalisation to unseen sizes. It should be cited as the canonical attention-over-nodes RL policy, not as evidence of size transfer.

### [hu2021updet] UPDeT (abstract_only; S2 API abstract)
- Argues that MARL models are usually retrained for every task because their input and output dimensions are fixed. For example, 3v3 and 5v6 games need different models.
- A transformer over observation entities decouples the policy from a fixed input layout. One architecture then serves tasks with different numbers of entities and transfers between them.
- Relevance: the direct MARL precedent for "one set of weights, varying entity count". Our per-token actor head is analogous to UPDeT's entity-wise action decoding.

### [yang2018mean] Mean Field Multi-Agent Reinforcement Learning (abstract_only; S2 API abstract)
- Approximates many-agent interaction by the interaction between each agent and the average effect of the population or its neighbours.
- Proposes mean-field Q-learning and actor-critic, with a convergence analysis. Tested on resource allocation, an Ising model and a battle game.
- Relevance: the population-statistic alternative to set architectures (theme d). It averages away the heterogeneity of individual EVs (departure time, energy demand), and per-EV constraint guarantees depend on exactly that heterogeneity.

## B. Size generalisation

### [joshi2022learning] Learning the TSP requires rethinking generalization (abstract_only; S2 API abstract; Constraints 2022)
- Neural TSP solvers that match classical solvers on small graphs fail to extrapolate to larger instances.
- A controlled pipeline study looks for the inductive biases, layers, learning paradigms and evaluation protocols that support zero-shot generalisation to larger sizes.
- Relevance: a warning that permutation-invariance alone does not guarantee size transfer. Distribution shift in local structure matters, so we should report performance as a function of N and not only at one large N.

### [luo2023neural] LEHD (abstract_only; S2 API abstract)
- A light-encoder, heavy-decoder model that re-attends to the remaining nodes at each step.
- Trained on small TSP/CVRP instances, it produces near-optimal solutions up to 1000 nodes and generalises to TSPLib/CVRPLib.
- Relevance: shows that attention models can transfer from small to large sizes when the architecture recomputes relations over the current set. Our encoder re-encodes the current charger set at each step in the same way.

## C. EV / energy papers

### [sadeghianpourhamami2020definition] Model-free coordination of EV charging with RL (full_text; https://arxiv.org/pdf/1809.10679)
- State (Sec. III-A/B, Alg. 1): the timeslot plus an S_max x S_max grid that counts connected EVs by time-to-departure and remaining charging time, normalised by N_max. The authors call it "scale-free" because its size does not depend on the number of stations.
- Action: one fraction per flexibility diagonal, giving the share of EVs in that class to charge. The mapping from fractions to individual EVs is not detailed.
- Infeasibility (EVs that can no longer finish) is penalised by M times their count (Eqs. 3-4), with M > 2N_max. Only feasible requests are assumed. There is no formal guarantee.
- Fitted Q-iteration on ElaadNL data. With 10 / 50 stations, cost falls 39% / 30.4% against business-as-usual, compared with 52% for the perfect-information optimum. The gap to the optimum is 13% / 15.6%.
- Generalisation (Sec. VI-C, Fig. 6): a policy trained with N_max = 10 is evaluated on session data scaled up to 10x. The largest cost increase occurs at 2x, and further scaling adds little. Exact values appear only in the figure.
- Relevance: the earliest EV size-agnostic RL design we found. It shows the aggregate-binning alternative to set architectures and reports gap to the optimum, the same metric we plan to use.

### [tuchnitz2021development] Smart charging strategy for an EV fleet based on RL (abstract_only; S2 API abstract)
- A central RL agent produces forward-looking schedules for "an electric vehicle fleet of any size", taking the grid baseload into account.
- Load variance falls 65% against uncontrolled charging, with performance close to an optimisation benchmark that needs departure times and energy in advance.
- The abstract reports that all EVs had enough energy at departure. This is an empirical outcome, not a guarantee.
- The abstract does not give training versus testing fleet sizes. The representation details need the full text, which was not accessible.

### [ye2022learning] Learning to operate an EV charging station considering VGI (CADE) (full_text; https://arxiv.org/pdf/2111.01294)
- Centralised allocation assigns EVs to charging or waiting spots. In decentralised execution, every charger acts with one DQN shared by all homogeneous chargers, trained from a shared replay memory (Alg. 2).
- Each charger's state combines its own EV's features with station summaries such as the number of waiting EVs and their remaining energy. Actions are discrete 1 kW steps within battery and power limits.
- Departure energy is a soft penalty on the shortfall (Eq. 14, mu = 0.2 $/kWh). The station peak appears only as a demand-charge cost. No guarantee is given.
- Sizes of 10, 20, 50 and 100 chargers are evaluated (Sec. IV-D, Table III), at 0.7-1.7 ms per step against 3-935 s for MPC. The text does not clearly say whether one trained model was reused across sizes.
- Relevance: the closest EV-station precedent to a shared per-charger head, but it has no attention between chargers and only soft constraints.

### [zhang2022transfer] Transfer DRL-based large-scale V2G coordination (full_text; https://arxiv.org/pdf/2210.07013; arXiv only)
- An aggregator-level PPO agent. The state is 24 h of load history, the aggregate SOC and the load variance. The action is a scalar aggregate power.
- Aggregate power is dispatched to EVs by an SOC-buffer rule, with per-EV power clipping (Eqs. 13-17). The paper does not say how the residual left after clipping is handled.
- Trained with 509 EVs. The policy is reported as transferable to 5,090-50,900 EVs with loads scaled 100x (Sec. V-A). Direct transfer and retraining are compared only at 509 EVs (Table II): load variance falls 97.37% against 97.46%.
- There are several internal numeric inconsistencies, and constraint enforcement at aggregate level is not explicit.

### [yan2022cooperative] Cooperative EV charging control via MARL (abstract_only; S2 API abstract)
- Each EV agent has a collective-policy model that approximates the other agents' consumption, plus an independent SAC learner on local observations.
- Fully decentralised execution. The authors claim scalability to many agents, but the abstract does not say whether trained weights were tested at another EV count.
- The goal is to avoid transformer overload. The abstract does not describe how constraints are handled.

### [wang2022stabilizing] T-MAAC: transformer MARL for voltage control (full_text; https://arxiv.org/pdf/2206.03721)
- The policy transformer takes buses as tokens. Attention is masked by zone adjacency in place of positional encodings, and an aggregation layer reads out at the agent's PV node (Sec. 4.1). Parameters are shared across agents, and an agent ID is appended.
- The critic is a transformer over (observation, action) tokens for each agent (Sec. 4.2).
- Each network is trained separately: 141-bus with 22 agents and 322-bus with 38 agents. No transfer across sizes is reported.
- Voltage limits enter only as a reward barrier, plus action clipping. Controllable rate rises, for example from 75.8% to 97.8% on 141-bus in spring.
- Relevance: an attention-over-entities policy in power systems that is not evaluated for size transfer.

### [jin2026telemetry] GRAS-AVC: zero-shot graph MARL for voltage control (full_text; https://www.mdpi.com/1424-8220/26/18/5752/htm, page truncated after Sec. 4.2)
- A permutation-equivariant shared actor (Prop. 1) with three edge-conditioned message-passing layers and a shared per-inverter head that outputs a droop prior plus a residual. Twin critics use mean/max pooling and handle variable graph size.
- Trained jointly on 7 feeders of 15-141 buses. The 322-bus feeder with 38 DERs is used zero-shot: no replay, gradients or checkpoint selection.
- Safety comes from training penalties and imitation, plus a deployment-time AC power-flow screen that picks between the actor and a droop fallback. The authors state this is not a multi-step safety certificate.
- Violating bus-time pairs fall from 7.455% under droop to 4.027% with the actor alone and 2.723% with the screen.
- Relevance: the closest power-system analogue to our claim of one set of weights across sizes with a safety mechanism. Its safety mechanism is a model-based runtime filter, while ours is a closed-form projection.

### [yan2026hierarchical] HPC-RL: hierarchical constrained RL for V2G (full_text; https://arxiv.org/html/2608.03409)
- The upper level is GRG-SAC. The policy outputs only the basic OPF variables, and the non-basic ones come from a Newton power-flow solve. Gradients pass through the implicit function theorem, and inequality constraints are handled by projection (Sec. 4.1).
- The lower level uses "dynamic boundary" per-EV bounds (Eq. 16): P_min = max{0, (SOC_d - SOC) - d_re P_phys}, P_max = min{P_phys, SOC_max - SOC}. These are the same kind of l_i/u_i bounds as our projection.
- Prop. 1 proves only first-order tangent-space feasibility. The paper claims demand satisfaction without proof and does not treat arrivals that cannot reach their target.
- IEEE 14, 30 and 141-bus cases. The objective gap to MPC is about 7-13%, demand satisfaction is 100%, and runtime is 0.87-4.97 s against up to 5173 s for MPC. There is no evaluation at a different EV count than training (Table 4 only times inference).
- Relevance: the strongest prior work on hard V2G constraints in RL. It is not size-agnostic and has no attention.

### [valesalonso2026emergent] Emergent charging coordination in electric delivery fleets (full_text; https://arxiv.org/html/2609.07689; arXiv only)
- One shared per-vehicle policy (NEAT or PPO). The per-candidate scorer is applied to each of up to 5 nearby stations, so it is invariant to the number of candidates.
- Trained on 4 cities and evaluated zero-shot on 16 unseen cities with 3-10 stations and about 21-138 vehicles. Fleet density was swept from 5 to 50 EVs per station without retraining (Fig. 11).
- The SoC floor and time budget are enforced only through the reward. There are no guarantees.
- Relevance: evidence that decentralised shared policies transfer across fleet sizes in an EV setting. The problem is routing and charging choice, not power control.

### [caballe2026largescale] Mean-field vs RL for large-scale EV charging (abstract_only; S2 API abstract)
- Compares mean-field games and mean-field-type games with classical and deep RL, using nine KPIs.
- Mean-field policies depend only on population statistics and are epsilon-Nash as fleet size grows. The test cases are a 3-node one-night grid and a 7-day testbed with 1,000 heterogeneous EVs.
- Mean-field-type games minimise unmet energy and queueing, while DRL reaches the highest average final SOC (about 81%).

### [hassouna2026graph] Graph RL for power grids: survey (abstract_only; S2 API abstract)
- Reviews graph RL for transmission and distribution grids by graph structure, GNN type and RL algorithm.
- Concludes that graph RL adapts to unpredictable events and noisy data but is still mostly proof-of-concept.
- Useful as an umbrella citation for GNN-based, topology-variable power-system RL.

## Theme synthesis

Making a learned controller independent of the number of controlled units rests on a small set of foundations. Permutation-invariant set functions [zaheer2017deep], self-attention [vaswani2017attention] and its set-specific forms [lee2019set] let a network process a variable-size collection of entities with shared parameters. In RL, the same idea gave attention-based policies over nodes for routing [kool2019attention] and entity-wise transformer policies that transfer across team sizes in MARL [hu2021updet]. The combinatorial-optimisation literature is a caution here. A permutation-invariant architecture is necessary for size transfer but not sufficient: policies trained on small instances often fail to extrapolate [joshi2022learning], and recomputing relations over the current set helps [luo2023neural]. A credible zero-shot claim should therefore report performance as a function of N, against an optimum, over the whole transfer range.

In EV charging and V2G, the dominant route to fleet-size independence has been aggregation, not set architectures. Examples are binned aggregate states with a fixed size regardless of the number of stations [sadeghianpourhamami2020definition], central agents for "fleets of any size" [tuchnitz2021development], aggregator-level actions dispatched by rules [zhang2022transfer], and mean-field formulations [yang2018mean], [caballe2026largescale]. These approaches do transfer across sizes: [sadeghianpourhamami2020definition] evaluates a 10-station policy on up to 10x more arrivals, and [zhang2022transfer] reports moving from 509 to 50,900 EVs. But they discard individual EV information or push it into hand-made disaggregation rules, so per-EV departure-energy feasibility is either penalised or left to heuristics. A second route is parameter sharing across chargers or EVs: a shared per-charger DQN [ye2022learning], decentralised per-EV SAC learners [yan2022cooperative], and shared per-vehicle policies that transfer zero-shot to unseen cities and fleet densities [valesalonso2026emergent]. None of these lets the chargers attend to each other. Each handles departure demand or the station limit with reward penalties, and [ye2022learning] does not clearly separate cross-size reuse from retraining.

Attention and graph policies with explicit size transfer are better developed in power-system voltage control than in EV charging. T-MAAC uses a transformer over bus tokens with shared actor parameters, but trains each network separately [wang2022stabilizing]. GRAS-AVC trains one permutation-equivariant graph actor on 15-141-bus feeders and deploys it zero-shot on a 322-bus feeder [jin2026telemetry]. Its safety comes from a deployment-time AC power-flow screen with a fallback, which the authors say is not a formal multi-step guarantee. The survey [hassouna2026graph] likewise describes graph RL for grids as mostly proof-of-concept. On hard constraints, the strongest V2G work, HPC-RL [yan2026hierarchical], combines power-flow-embedded projection with per-EV dynamic energy bounds of the same form as our l_i/u_i. It proves only a first-order feasibility property, gives no proof for demand satisfaction, and is not evaluated at a different fleet size than it was trained on.

Across the screened literature (EV-specific GNN/decision-transformer works such as EV-GNN, GNN-DT, PI-TD3 and HetGPS are reviewed in other parts), we found no paper in L3 that combines a size-agnostic attention policy, a provable per-step guarantee for both EV departure energy and the station limit, and a reported optimality gap under zero-shot transfer across charger counts. The nearest pieces are scattered across three papers: size-agnostic graph policies with heuristic safety screening [jin2026telemetry], hard V2G constraints with partial proofs and fixed size [yan2026hierarchical], and aggregate or size-independent EV representations evaluated at larger scales without hard guarantees [sadeghianpourhamami2020definition], [zhang2022transfer].
