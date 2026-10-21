# L4 notes: method-section references (conformal prediction, departure prediction, architecture/training, solvers, statistics)

Agent A1, part L4, 2026-10-22. Read levels follow `L4_matrix.csv`. `metadata_only` means that only the Crossref record was checked. Any claim about such a work comes from a secondary source that was read, and that source is named.

## 1 Conformal prediction foundations

### [lei2018distributionfree] Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, JASA 113(523):1094-1111, 2018 (full text, arXiv 1604.04173)
- Sec. 2.2, Alg. 2 (split conformal): the data are split into two halves. A regressor is fit on one half and absolute residuals are computed on the other. The interval half-width is the k-th smallest residual, with k = ceil((n/2+1)(1-alpha)).
- Thm 2.2: under i.i.d. data, P(Y_{n+1} in C) >= 1-alpha. If the residuals have a continuous joint distribution, coverage is also at most 1-alpha+2/(n+2). The upper bound means the method is not overly conservative.
- The coverage is marginal, averaged over the calibration and test draws. It is not conditional on x.
- Sec. 2.2 credits inductive conformal inference to [papadopoulos2002inductive] and Vovk et al. (2005); see [vovk2022algorithmic] for the 2nd edition.
- Thm 2.3 gives approximate in-sample coverage for split conformal.
- Use for us: this is the standard reference for split conformal and the rank/quantile argument behind our Proposition 2. Our lower bound is one-sided. It uses the ceil((n_cal+1)(1-alpha))-th order statistic, the same rank argument applied to a one-sided score.

### [romano2019conformalized] Romano, Patterson, Candes, NeurIPS 2019 (full text, arXiv 1905.03222)
- CQR (Alg. 1) fits lower and upper quantile regressors on I1. The conformity score is E_i = max(q_lo(X_i)-Y_i, Y_i-q_hi(X_i)) on I2. The interval is widened (or shrunk) by the (1-alpha)(1+1/|I2|) empirical quantile of the scores.
- Thm 1: under exchangeability, P(Y in C(X)) >= 1-alpha. Thm 2 is an asymmetric version that conformalizes each tail separately. In Sec. 6 it gives slightly longer intervals.
- Sec. 6: uncalibrated quantile neural networks can under-cover. Conformalizing restores nominal 90% coverage on 11 benchmark datasets.
- Use for us: our predictor is a quantile regressor (quantile gradient boosting) for the departure step, and conformal calibration of its lower quantile is the one-sided case of CQR. Cite it for the "quantile model + conformal correction" design.

### [angelopoulos2023conformal] Angelopoulos and Bates, FnT ML 16(4):494-591, 2023 (abstract + Sec. 1 read)
- A tutorial. Conformal sets are valid in a distribution-free, finite-sample sense and work with any pre-trained model.
- It states clearly that the guarantee is marginal, and that conditional coverage is impossible in general (Sec. 1, conditional-coverage paragraph). This is a good citation for the limitation in our Proposition 2: the guarantee holds on average over EVs, not for each individual EV.

### [vovk2022algorithmic] Vovk, Gammerman, Shafer, Springer, 2nd ed. 2022 (metadata only)
- The canonical monograph. The 1st edition (2005, DOI 10.1007/b106715) has an author-less Crossref record, so the 2nd edition was verified instead. Its content was not read here; it is cited as the origin of conformal prediction, as in [lei2018distributionfree] and [angelopoulos2023conformal].

### [papadopoulos2002inductive] Papadopoulos, Proedrou, Vovk, Gammerman, ECML 2002, LNCS pp. 345-356 (metadata only)
- The original inductive (split) conformal regression. The Springer page could not be reached. Its role is confirmed by [lei2018distributionfree], Sec. 2.2.

## 2 Conformal prediction in EV/energy control and in safe control or RL

### [lindemann2023safe] Lindemann, Cleaveland, Shim, Pappas, IEEE RA-L 8(8):5116-5123, 2023 (full text, arXiv 2210.10254)
- Sec. 3: the nonconformity score is the norm of the prediction error. C is the ceil((k+1)(1-delta))-th smallest calibration score. Only exchangeability is required.
- Thm 1 gives conformal regions for multi-step trajectory predictions. For all horizons at once it uses a union bound (delta/H per step).
- Thms 2-3: an MPC with constraints tightened by the conformal regions is safe with probability at least 1-delta. The open-loop version is Thm 2 and the closed-loop version, which needs a recursive-feasibility assumption, is Thm 3.
- Sec. 5 (CARLA, pedestrian intersection): in 100 MPC runs the 95% guarantee was confirmed empirically.
- Use for us: this is the closest template for "conformal bound + hard-constraint controller gives a probabilistic guarantee". Our Proposition 2 has the same structure: a deterministic per-step guarantee conditional on the bound holding, which yields P(served) >= 1-alpha.

### [fernandezzapico2025stochastic] Fernandez-Zapico, Hofman, Salazar, IEEE CDC 2025, pp. 3095-3100 (full text, arXiv 2504.00685)
- An EV charging hub with PV and a battery. Scenario-based stochastic MPC decides battery (dis)charging.
- Forecasts come from gradient-boosted trees (Eq. 12). Conformal intervals come from EnbPI via MAPIE (Eq. 13).
- EV load is modelled in aggregate (Eqs. 7-9). Per-EV connection and disconnection steps are used only to build the load; departure is not predicted per EV.
- Tab. II: coverage of about 0.90 at alpha = 0.1 for EV load, but poor coverage for prices in autumn (0.22).
- Result: 13% more expensive than perfect-forecast MPC and 1% cheaper than point-forecast MPC, over a 280-day test.
- Takeaway: conformal is used in EV charging control, but on aggregate forecasts. There is no per-EV departure bound and no RL.

### [cheng2026riskaware] Cheng, Yan, Nan, Sun, Energies 19(17):4093, 2026 (abstract only)
- A one-sided conformal upper bound on net load (EV load minus PV), built from forecast residuals. The calibration parameter is adapted online from miscoverage events (adaptive conformal). The bound is used in rolling-horizon MPC for a PV-battery-EV station.
- Compared with fixed conformal and point-forecast MPC, it gives better event-period coverage and less low-SOC exposure (abstract-level).
- Again an aggregate bound, not a per-EV departure bound.

### [zhou2025computationally] Zhou, Zhang, Luo, IEEE ICRA 2025 (abstract; PDF Sec. I-III skimmed)
- Safe episodic RL and control: a GP dynamics model (quadrature Fourier features), adaptive conformal prediction to quantify model error, and discrete-time control barrier functions tightened by the conformal bound, with optimistic exploration.
- It claims provable safety and sample efficiency. Results are in simulation only.
- Use for us: an example of conformal bounds feeding a safety filter in RL. Our use is different: the bound is on an exogenous quantity (departure time), not on the dynamics.

### [phipps2023customized] Phipps, Schwenk, Briegel, Mikut, Hagenmeyer, IEEE IoT J. 10(23):20649-20661, 2023 (abstract only)
- Uncertainty of parking-duration prediction for user-centric smart charging. Errors are split into a critical part (over-predicted duration, which leads to undercharging) and a non-critical part.
- Quantile-based "security levels" chosen per user risk preference reduce the probability of a critical error.
- Four probabilistic forecasters are tested on a semi-synthetic open mobility dataset and on real EV trips. Critical errors are regulated even on noisy real data.
- The abstract mentions neither conformal calibration nor a finite-sample guarantee, nor integration into a controller. The full text (IEEE OA PDF, TechRxiv) was not reachable from the sandbox, so this must be checked before submission.
- **This is the closest prior work to our secondary contribution.** We should cite it and position against it. We add finite-sample conformal calibration, a one-sided lower bound on departure used inside a projection layer, and an end-to-end P(served) >= 1-alpha statement.

## 3 EV session duration and departure prediction

### [lee2019acndata] Lee, Li, Low, ACM e-Energy 2019, pp. 139-149 (abstract only)
- Releases ACN-Data (workplace charging; more than 30,000 sessions at release). It uses Gaussian mixture models to learn and predict user behaviour, plus solar-sizing and duck-curve examples.
- Use for us: the source of EV2Gym-style session statistics, and an early learned model of session parameters. Note: the Crossref title is only "ACN-Data", while the BibTeX carries the full title.

### [lee2026enabling] Lee, Hwang, Kondoro, Song, Noh, AAAI 2026 (full text, arXiv 2512.07723)
- Departure prediction is framed as discrete-time survival analysis (5-min interval tokens) with a Transformer over streaming smartphone context.
- Data: a real-world study with 93 users (2021-2022). Baselines include SVR, LightGBM, FT-Transformer and sequence models.
- Tab. 1: MAE of 2.20 h, 14.4%/15.1% better than the best baselines. The ablation improves this to 2.13 h.
- The setting is home departure, aimed at delayed-full charging for battery life. The model outputs survival probabilities but gives no calibrated coverage guarantee.
- [frendo2021improving], already in references.bib, is the classic station-side departure-prediction paper. [phipps2023customized] covers parking duration with uncertainty.

## 4 Architecture and training details

- [xiong2020layer] (ICML 2020; abstract + Sec. 1): a mean-field analysis shows that the Post-LN Transformer has large gradients near the output at initialisation and therefore needs learning-rate warm-up. Pre-LN has well-behaved gradients, so warm-up can be removed with comparable results. This justifies our pre-LN encoder and the absence of warm-up.
- [ba2016layer] (abstract): LayerNorm normalises over the features of a single sample and does not depend on batch size. This suits variable-N token sets with padding (statistics are computed per token).
- [schulman2016highdimensional] (abstract): GAE is an exponentially weighted sum of TD residuals, analogous to TD(lambda), and trades bias against variance. We use lambda_GAE = 0.95. The bib year is 2015 (arXiv); the venue is ICLR 2016.
- [kingma2015adam] (abstract): Adam uses adaptive moment estimates. The bib year is 2014 (arXiv); the venue is ICLR 2015.
- [huang2022cleanrl] (abstract + Sec. 1): single-file DRL implementations. Our PPO follows its style. The JMLR venue could not be confirmed (S2 lists arXiv only). The arXiv author metadata lists "Chang Ye", and the generated BibTeX should be checked.
- [paszke2019pytorch] (abstract): the implementation framework.
- "The 37 Implementation Details of PPO" (ICLR Blog Track 2022) has no DOI or arXiv ID, so it is skipped.

## 5 Solvers
- [huangfu2018parallelizing] (abstract; MPC 10(1):119-142): parallel dual revised simplex (PAMI/SIP). This is the standard citation for HiGHS, which is used for the perfect-information LP oracle and the admission LP.
- [diamond2016cvxpy] and [bynum2021pyomo] are verified but optional. Cite whichever modelling layer the code actually uses. The current `algorithm.md` uses `highspy` directly, in which case neither is needed.

## 6 Statistical evaluation
- [agarwal2021deep] (abstract + Sec. 1; NeurIPS 2021, pp. 29304-29320): point estimates from few runs are unreliable. It recommends stratified-bootstrap confidence intervals, performance profiles and the interquartile mean (IQM); IQM has much smaller CIs than the median. Use for reporting across seeds and test days.
- [henderson2018deep] (abstract; AAAI 2018): deep RL results vary strongly with seeds and implementations, and significance testing and standardised reporting are needed.
- [wilcoxon1945individual] (metadata only; Biometrics Bulletin 1(6):80-83): the original source of the signed-rank test, for paired per-episode comparisons.
- Holm (1979), Scand. J. Statistics 6(2):65-70, has **no DOI** (JSTOR stable 4615733 only). It cannot be verified with `verify_ref.py`, so it is not added. The orchestrator should decide whether to add it manually as a classic, or to state "Holm-Bonferroni correction" with a citation through a verified secondary source.

## Theme synthesis

Split (inductive) conformal prediction [papadopoulos2002inductive; vovk2022algorithmic; lei2018distributionfree] turns any point or quantile predictor into prediction sets with finite-sample, distribution-free marginal coverage. The only assumption is exchangeability between the calibration data and the test point. Conformalized quantile regression [romano2019conformalized] keeps the adaptivity of a quantile model and restores the nominal coverage that uncalibrated quantile models often miss. The guarantee is marginal, not conditional [angelopoulos2023conformal; lei2018distributionfree], and our Proposition 2 inherits exactly this property. It bounds the fraction of EVs whose departure precedes the conformal lower bound, not the risk for a particular EV.

In control, conformal regions have been used to tighten constraints of model-predictive controllers, which turns a deterministic safety argument into a probabilistic one [lindemann2023safe]. They have also been used to tighten control-barrier-function constraints in safe RL [zhou2025computationally]. In EV charging, the conformal work found here addresses aggregate quantities: EV-hub load, PV and prices in a stochastic MPC [fernandezzapico2025stochastic], and a one-sided adaptive conformal net-load bound in rolling-horizon MPC [cheng2026riskaware].

Prediction of EV departure time or parking duration is established, from user-behaviour models on ACN-Data [lee2019acndata] and departure prediction for charging prioritisation [frendo2021improving] to survival-type Transformers for home departures [lee2026enabling]. Only [phipps2023customized] explicitly shapes the uncertainty of parking-duration predictions around the undercharging risk, using quantile-based security levels. From its abstract, it neither uses conformal calibration with a finite-sample guarantee nor embeds the bound in a charging controller.

Novelty check (secondary contribution): in this search (arXiv full-text search, Crossref bibliographic search; Semantic Scholar search was rate-limited and OpenAlex search had no remaining budget; web search was unavailable), no paper was found that uses a conformal lower bound on each EV's departure time inside an EV charging controller to obtain an energy-delivery guarantee. The same holds for RL-based charging controllers and safety layers. The defensible claim is: "to our knowledge, the first combination of a split-conformal departure lower bound with a constraint-guaranteed RL action layer, giving P(EV served) >= 1-alpha". It should cite [phipps2023customized] as the closest work (quantile-based, not conformal; prediction only) and [lindemann2023safe] as the methodological template.
