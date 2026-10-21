# L1 notes: RL for EV smart charging / V2G, plus scheduling context

Read levels: FT = full text (via WebFetch summaries of arXiv/MDPI pages), AO = abstract only.
For every paper, four questions are answered explicitly: (i) departure-energy constraint, (ii) power/transformer constraint, (iii) guarantee claimed, (iv) fixed vs variable number of chargers, plus simulator/data.

---

## A. Deep RL for station-level EV charging / V2G

### [sadeghianpourhamami2020definition] (FT, arXiv 1809.10679v2)
- Joint control of a set of charging stations with batch RL (fitted Q-iteration). The state is a 2-D grid that bins EVs by time-to-departure and remaining charging time, normalised by N_max. Its size depends only on the number of time slots, not on the number of stations (Sec. III-A, Alg. 1, Fig. 1).
- Action: for each bin diagonal (same flexibility), the fraction of EVs charged in the next slot (Sec. III-B).
- (i) Departure energy: soft penalty M x (number of EVs that can no longer finish), with M > 2 N_max (Eq. 3/4). Requests are assumed feasible, and violation rates are not reported.
- (ii) No transformer/power limit. The objective is load flattening via a quadratic cost on total power (Eq. 2). (iii) No guarantee. (iv) Variable N by construction.
- Data: ElaadNL (busiest 10/50 stations, 2015). Oracle: a perfect-knowledge quadratic optimisation. The learned policy costs 13% (10 stations) and 15.6% (50 stations) more than the oracle. Cost reduction vs BAU is 39% and 30.4% (Sec. VI-A).
- Generalisation: trained on 10 stations and tested with arrivals scaled up to 10x. The biggest cost jump comes at 2x, with only marginal increases afterwards (Sec. VI-C, Fig. 6).
- Relevance: the closest prior precedent for contributions 1 and 3. It already gives a scale-independent representation, a train-small/test-large evaluation and an oracle-gap metric. It differs from ours because it uses an aggregated binned state (not per-charger tokens), discrete fractional actions, no prices/V2G, no transformer limit and only a penalty for energy.

### [korkas2024distributed] (FT partial, MDPI HTML)
- Station with N spots, PV and V2G. There is one DDPG agent per spot, trained with difference rewards computed by a global coordinator (Sec. 3.2-3.3).
- The local observation stays at 10 values regardless of station size (10/20/35/50 spots). Centralised input grows to 108 values at 50 spots (Sec. 3.1). The paper claims spots can be added "on the fly". Weight sharing and zero-shot transfer across sizes are not stated in the part we read.
- (i) Departure energy: squared penalty on (1 - SoC) of departing EVs (Eq. 7). (ii) Per-vehicle power limits only, with no aggregate station/transformer cap (Eq. 2-3, 6). (iii) No guarantee. (iv) Variable N (one agent per spot).
- Environment: Chargym, four price profiles (incl. NordPool), a typical PV day. Baselines: RBC and centralised DDPG/SAC. Abstract: up to 25% cost reduction and up to 20% better user satisfaction.
- Relevance: a per-charger decomposition for scalability, but with no attention, no hard constraints and no transformer limit.

### [zhang2024hamlet] (AO)
- Charging power control of a *dynamic number* of EVs in a distribution feeder (voltage, bill, comfort). Hierarchical MDP with multi-objective RL.
- Variable N is handled by grouping EVs with the same remaining charging time into one agent, which is a state/action aggregation.
- (i) Comfort/anxiety reward distributed over the charging interval, i.e. a penalty. (ii) Voltage handled via the objective. (iii) No guarantee stated. (iv) Variable N.
- IEEE 33/69-bus tests.
- Relevance: an alternative answer to variable N (grouping rather than tokenisation).

### [zhou2025optimal] (AO)
- A station with smart-charging and uncontrolled-charging users. Decentralised DDPG per charging unit, with an energy-boundary model plus a relaxation mechanism to keep the schedule aligned with demand.
- Centralised training of the decentralised units is said to give consistent performance across multiple charging units with fewer training resources. This is a scalability claim based on shared training.
- (i) Energy-boundary model with relaxation (details not read). (ii) Not stated. (iii) None stated. (iv) Variable N (per unit). Survey data from Jiangsu.

### [song2025longterm] (AO, SSRN page)
- MAHEM: a transformer-based MARL for multi-station collaborative charging. The transformer captures long-term temporal features, agents communicate through attention, and Q-value decomposition shrinks the action space.
- Reports a 31.6% cost reduction vs baselines and claims robustness and transfer tests.
- (i)/(ii) Constraint handling not stated in the abstract. (iii) None. (iv) The abstract gives no evidence of a variable number of chargers.
- Relevance: the closest "transformer + RL + EV charging" paper found in this theme. Its attention runs over time and agents (stations), not over chargers as tokens with a shared per-token head. Needs a full-text check (see threats).

### [yilmaz2024reinforcement] (AO)
- CPO perspective, commercial parking lot. DDPG minimises power-setpoint tracking error.
- (ii) Power constraints are incorporated "by structuring the action and state space". The mechanism is not read. (i) Varying demands are handled, but how is not stated. (iii) None stated. (iv) Multiple chargers; fixed size presumably.
- Same group as EV2Gym (Orfanoudakis, Vergara), so the setting is very close to ours.

### [naghdizadeganjahromi2026adaptive] (AO)
- PPO for EV participation in the Belgian imbalance market (2023 data), at an office parking facility.
- Explicitly models departure-time uncertainty. Key finding: policies trained *without* departure uncertainty outperform those trained with stochastic departure (aggressive arbitrage beats conservative modelling).
- 1-minute price noise cuts flexibility volume by 5% but returns by less than 1%.
- (i)/(ii) Constraint mechanism not in the abstract. (iii) None. (iv) Fixed (single facility).
- Relevance: supports our assumption of known departure times, and shows a cost of conservative uncertainty modelling.

### [ponse2025chargax] (FT, arXiv 2507.01522)
- JAX charging-station environment (tree from grid connection to EVSEs). Configurable chargers (16 in main tests). PPO via PureJaxRL.
- (ii) Architecture/power limits are enforced by rescaling leaf currents, a projection-like correction (App. A.2, Eq. 5), or optionally by a soft penalty. (i) Unmet demand is handled by satisfaction penalties (App. A.3). (iii) No guarantee. (iv) Configurable, but the observation concatenates per-EVSE states, so the policy is tied to station size.
- Speedups over EV2Gym, Chargym and SustainGym range from 17x to 2820x depending on setting (Table 2). Limitations: isolated network, no shared transformer, no local generation.

### [arif2026constraint] (AO, SSRN preprint, not peer reviewed)
- Bus-depot charging on a weak 11 kV feeder, with two vehicle pools and separate deadlines. A 6,165-parameter DQN.
- (ii) The network constraint is enforced by a *closed-form projection* applied to the action before it reaches the plant. There are zero voltage-floor breaches across 38 trained networks (8 seeds, 19 hyperparameter settings), 40 test days.
- (i) Deadlines are part of the CMDP. Without the projection layer, reward shaping cuts violations to zero but leaves 34 kWh undelivered. (iii) Empirical, not proved (per abstract).
- Costs 45.0% less than uncoordinated charging and 4.4% above a perfect-foresight LP bound.
- Main message: once a hard projection layer is present, constraint penalties carry no gradient and are redundant. This directly supports our design choice (contribution 2) and partly overlaps with it.

### [li2020constrained] (AO)
- Single-EV charging/discharging under random arrival/departure and prices, framed as a CMDP. Model-free safe DRL is used, so no penalty coefficient has to be hand-tuned.
- (i) The "fully charged at departure" requirement is a CMDP constraint. The abstract says "guarantee", but CMDP methods give constraint satisfaction in expectation, so we treat it as empirical. (ii) No station limit (single EV). (iv) Single EV.
- Relevance: the standard Lagrangian/CMDP comparator for our projection layer.

---

## B. Reviews

### [qiu2023reinforcement] (AO)
- Critical review of RL for EV dispatch (G2V, V2H, V2G), organised by single-agent vs multi-agent RL.
- Stated challenges: data quality and availability, environment setup, safety and robustness, training performance (and more; the abstract is truncated).
- Useful as the general citation for "RL for EVs in power systems".

### [michailidis2025reinforcement] (FT partial, MDPI HTML)
- Reviews EV charging station RL from 2015-2025. More than 200 papers were screened, and citation thresholds were relaxed for 2022-2025 (Sec. 2).
- Departure demand usually enters as reward penalties. Transformer/charger limits are mentioned as "physical constraints" without a mechanism (Sec. 4.1). Safe/constrained policy methods are mentioned for policy-based RL (Sec. 6.1). Projection/action correction is not discussed in the accessible text.
- Common baselines: rule-based/TOU, LP/MILP offline scheduling, MPC, greedy (Secs. 1.1, 5.1).
- Gaps: training instability, safety and explainability, heavy reliance on simulation, no standardised EVCS benchmark, multi-agent non-stationarity. Transfer learning is named as a future direction (Sec. 6.1).
- Caveat: table bodies and Secs. 7-8 were not accessible.

---

## C. Laxity / deadline scheduling

### [chen2022smoothed] (FT, arXiv 2102.08610)
- Adaptive charging is posed as a feasibility problem: energy by deadline, per-EV rate limits and a station power limit P(t) (Eqs. 1-3).
- Laxity l_i(t) = [d_i - t]^+ - e_i(t)/rbar_i. Prop. 1: a schedule is feasible if and only if all laxities stay non-negative (Def. 4, Sec. II-C).
- sLLF maximises a concave sum of next-slot laxities. Its solution is a closed form: r_i = [rbar_i (L(t) - l_i(t) + 1)] clipped to [0, min(rbar_i, e_i)]. The common level L(t) is found by bisection so that total power equals min(P, sum of caps) (Eq. 16-17, Prop. 2).
- Guarantees: offline feasibility (Cor. 1) and resource-augmentation theorems (Thm 1-2, Cor. 3: about 3-power feasible). Empirically, it needs only 7% extra power for 100% feasibility, vs 139% for EDF and 28% for online LP (Table III).
- Assumes departure time and energy are known at arrival (ACN user input). The objective is feasibility only, with no prices or V2G.
- Relevance: **the most important precedent for contribution 2**. Our p_i = clip(a_i - lambda, l_i, u_i) has the same algebraic form (per-EV clip around a common shifted level found by 1-D search). Our differences: the projection acts on a learned action, it is a differentiable layer inside RL training, it includes a lower bound l_i for the departure guarantee, and it allows bidirectional (V2G) bounds.

### [hao2022laxity] (AO)
- Public station with joint online pricing and charging scheduling. A laxity-differentiated pricing (LDP) scheme is combined with a deadline-differentiated threshold charging (DTC) policy, which the authors show to be optimal under arbitrary dynamics.
- The threshold policy is embedded in SAC to reduce action dimensionality.
- (i)/(ii) Handled via the structure of the threshold policy (details not read). (iii) Optimality of the policy *structure* is proved, but the RL part is not guaranteed. (iv) Variable number of EVs via the structured policy.
- Relevance: laxity and deadline structure used to make RL scale. A related SAC + nodal-multi-target paper (10.1109/TSG.2020.3028470, TSG 2021) was screened but not included because of the 20-paper cap and the share of pre-2022 papers.

### [vanhuffelen2025grid] (FT, arXiv 2403.03109)
- Online scheduling for a network of parking lots under cable capacities, minimising tardiness. EVs always get their full energy but may leave late (Sec. 2).
- Uses online serial/parallel schedule-generation schemes with FCFS/EDD/ELSTu (latest start, laxity-like) priorities plus destroy-and-repair (Sec. 3). Capacity is enforced by path-wise feasibility checks.
- Utrecht data (about 1,125 EVs/day). Uncontrolled charging overloads cables 60-70% of the time. The best method gives about 95 s average delay and 1.6% of EVs delayed (Tables 3-4). EDD is close behind; FCFS is much worse.
- Relevance: shows that deadline/laxity priority rules are strong, cheap baselines under hard capacity limits.

---

## D. Departure-time / user-behaviour uncertainty

### [frendo2021improving] (AO)
- Workplace charging with scarce power. Regression models predict EV departure from historical data, and the predictions drive a prioritising heuristic. Better predictions give better fair share.
- (i) Fair-share prioritisation, not a hard constraint. (ii) Infrastructure supply limit. (iii) None. (iv) Variable.

### [dyo2022behavior] (FT, arXiv 2202.10823)
- Shows that domestic plug-in duration is hard to predict: linear regression gives MAE about 14 h and MAPE about 414%, and DNNs gain little (Sec. III, Table I).
- Proposes a boost-then-slow policy whose parameters are learned retrospectively per charger. This gives a 31% aggregate evening-peak reduction on 3.17M UK sessions. The energy deficit is 1.4-5% depending on history length (Sec. V).
- (i) Energy loss enters as a penalty (reward goes to minus infinity beyond 10 kWh). (ii) No station limit. (iii) None. (iv) Single charger.
- Relevance: evidence that the "known departure time" assumption is strong for residential settings. It motivates stating our assumption explicitly and mentioning prediction/robust variants as future work. No conformal-prediction EV-departure paper was found among the peer-reviewed records screened.

### [naghdizadeganjahromi2026adaptive]: see Section A (stochastic departure in training).

---

## E. MPC and optimal benchmarks

### [lee2021adaptive] (FT, arXiv 2012.02636)
- ACN architecture, deployed since 2016. The Adaptive Scheduling Algorithm solves a convex program in receding horizon, re-solving on events or every 5 min (Sec. III-B, V).
- (i) Energy delivered is constrained to be at most e_i (5c), with a non-completion penalty. The paper notes the zero vector is always feasible, so delivery is best effort. (ii) Hard infrastructure constraints (SOC transformer/phase limits or a conservative affine version, Eqs. 1-2, 5d). (iii) No guarantee on energy delivery. (iv) Variable N.
- Results: profit about 3.4x uncontrolled; 98.1% of the offline optimum (idealised) and about 90% (non-ideal). ASA-QC is within 0.4% of optimal energy delivery and beats EDF by up to 14.1% at 30 kW. Baselines: LLF, EDF, RR (Sec. VI).
- Relevance: the reference MPC baseline with an offline-optimal comparison and the same "fraction of offline optimum" metric we use.

### [li2021learning] (FT, arXiv 2012.11261)
- Aggregator/operator split. The aggregator learns the maximum entropy feedback (MEF) with SAC, and the operator runs penalized predictive control (PPC) (Secs. III-V).
- (i) Energy delivery and tracking are reward penalties during MEF learning. Per-EV rate and arrival constraints are enforced by a fixed LLF dispatcher (Sec. VI-A.2). (ii) Aggregate power levels. (iii) Thm V.1 (provable): PPC with the *exact* MEF reproduces offline optimal actions under strict convexity/log-concavity. With the learned MEF, results are only "almost" feasible. (iv) Fixed (54 chargers).
- Data: ACN-Data, CAISO LMPs. Baselines: MPC with full state, offline optimal.
- Relevance: an example of learning combined with a provable statement under idealised assumptions, and of LLF as a dispatch layer under an RL decision.

### [diazlondono2026open] (FT of the preprint arXiv 2405.11963, same authors; the published EPSR version was not read)
- eMPC and OCMF formulations for G2V (LP) and V2G (MILP) at a CPO, integrated with EV2Gym (Alg. 1).
- (i) Departure SoC is a hard bound in the MPC (Eq. 10, min 80%). (ii) Transformer limit with demand-response reduction (Eq. 14). (iii) Hard constraints within the optimisation, with known EV schedules assumed. (iv) Variable (5-60 EVSEs in timing tests).
- 10 EVSEs, about 25 EVs, 50 runs: eMPC V2G profit 50.8 EUR vs AFAP -47.6 EUR, with higher battery degradation. Worst-case solve time is under 1.5% of the time step (Sec. V).
- There is no perfect-information benchmark, which leaves room for our optimality-gap reporting.

---

## Theme synthesis

Deep RL for EV charging has moved from single-EV agents with price signals [li2020constrained] to station-level and multi-station control with V2G, renewables and grid objectives [korkas2024distributed; zhang2024hamlet; song2025longterm; naghdizadeganjahromi2026adaptive]. The reviews agree that the field still depends on reward engineering for user demand and on simulation, and that it lacks standard benchmarks and safety guarantees [qiu2023reinforcement; michailidis2025reinforcement]. Across the papers read, the departure-energy requirement is almost always a reward penalty [sadeghianpourhamami2020definition; korkas2024distributed; ponse2025chargax; dyo2022behavior] or a CMDP constraint satisfied in expectation [li2020constrained]. Station or transformer limits are either absent [sadeghianpourhamami2020definition; korkas2024distributed], penalised, or enforced by ad-hoc rescaling of currents [ponse2025chargax]. None of the RL papers read states a formal per-step guarantee for both the departure energy and the station limit.

Scalability to the number of chargers is handled in three main ways. The first aggregates the state into a fixed-size representation, such as binning by time-to-departure [sadeghianpourhamami2020definition] or grouping EVs with equal remaining time [zhang2024hamlet]. The second decomposes the problem into per-charger agents with fixed local observations [korkas2024distributed; zhou2025optimal]. The third reduces the action space through structural optimality results (threshold/deadline policies) [hao2022laxity]. Attention appears mainly for temporal modelling and inter-station communication [song2025longterm], not as per-charger tokens with a shared actor head. Sadeghianpourhamami et al. already evaluated a policy trained on 10 stations under up to 10x more arrivals and reported the gap to a perfect-knowledge oracle (13-15.6%) [sadeghianpourhamami2020definition]. Our contribution 3 must therefore be framed as zero-shot transfer of a per-charger policy to about 200 *chargers* in a V2G, price-driven, transformer-limited setting, not as the first train-small/test-large study.

On the scheduling side, laxity is the central concept. A schedule is feasible if and only if all laxities stay non-negative, and smoothed LLF computes rates in closed form as a clipped water-filling around a common level found by bisection. It needs only 7% extra power to reach full feasibility on real traces [chen2022smoothed]. Deadline and latest-start priority rules remain strong, cheap baselines under hard network capacity [vanhuffelen2025grid; lee2021adaptive], and LLF is used as a dispatch layer beneath learned or predictive decisions [li2021learning]. A recent preprint shows empirically that a closed-form projection onto the network constraint gives zero violations and makes penalty shaping redundant in a DQN depot controller [arif2026constraint]. Together with sLLF, this means the *form* of our projection layer is not new. The novelty has to rest on combining it with a learned per-charger policy as a differentiable layer used in training, on the joint lower bound that guarantees departure energy under V2G, and on the stated feasibility proof.

Departure-time uncertainty is the main weak point of the "known departure" assumption. Residential plug-in durations are hard to predict (MAPE about 414%) [dyo2022behavior], while workplace predictions do improve prioritisation [frendo2021improving]. Interestingly, PPO policies trained without departure uncertainty outperformed those trained with stochastic departures in an imbalance-market setting [naghdizadeganjahromi2026adaptive]. For baselines, MPC with hard departure and transformer constraints is now available in EV2Gym [diazlondono2026open], and ACN reports performance as a fraction of the offline optimum (98.1%/90%) [lee2021adaptive]. These two works support our choice of MPC plus a perfect-information optimum as reference points, and the EV2Gym MPC study itself reports no optimality gap.
