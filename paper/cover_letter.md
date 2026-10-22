# Cover letter (draft)

*SAUCIS does not require a cover letter in its author guidelines; the DergiPark submission form may offer a field for one. SAUCIS does not ask for highlights, so none are provided. Fill the bracketed fields; do not add result claims before the experiments are complete.*

---

[Date]

To the Editor-in-Chief
Sakarya University Journal of Computer and Information Sciences (SAUCIS)

Dear Editor,

We submit the manuscript "Attention-Based Safe Reinforcement Learning with a Feasibility-Preserving Safety Layer for Size-Agnostic V2G Charging Control" [TODO: final title, chosen by the authors] for consideration as a research article in SAUCIS.

Charge point operators that offer vehicle-to-grid services must respect a transformer limit and deliver the energy each driver requests by departure. Reinforcement learning controllers for this task usually encode both requirements as reward penalties, and their networks are tied to one station size. The manuscript combines an attention policy, in which every charger is a token and one set of weights serves any number of chargers, with a safety layer that projects the raw actions onto taper-aware per-vehicle bounds and the transformer limit. The main theorem has two layers. Under stated simulator and grid-headroom conditions, the transformer limit holds at every step for every sequence of raw actions. The departure requirement is met for every vehicle whose stay contains no aggregate-infeasible step, and the paper measures how often such steps occur. The paper states openly that the projection has a known algebraic form and that each ingredient exists in some form, and it places its contribution in the combination, the two-layer result and its measured cost.

The evaluation trains on stations with 10 and 20 chargers in the open EV2Gym simulator and tests without retraining on stations with up to 200 chargers and on a tighter transformer rating. All methods receive the same admitted requirements, and the share of vehicles whose stay contains no infeasible step is a pre-declared primary endpoint. The main metric is the gap of a penalised profit to a perfect-information optimum that bounds every compared method, and a hard-constrained optimum measures the cost of the guarantee. Baselines include a penalty-trained twin, a Lagrangian constrained-RL method, a Deep Sets policy, rule-based schedulers, a receding-horizon model-predictive controller and the public EV-GNN policy. [RESULT: one sentence with the main finding, to be written by the authors after the experiments.]

The topic lies at the intersection of machine learning, optimisation and smart-grid applications. [TODO: optionally cite one or two related SAUCIS articles that the authors have checked.] We believe the paper fits the journal's scope of artificial intelligence and its applications.

The manuscript is original, has not been published before and is not under consideration elsewhere. All authors approved the submission. The code, scenario configurations and analysis scripts will be made public [TODO: repository link or "on acceptance"]. The use of an AI assistant (Claude, Anthropic) for literature work, method derivation, experiment design, code and text drafting is disclosed in full in the Artificial Intelligence Statement of the Article Information Form. The experiments were run by the authors, and the interpretation of the results was written by the authors.

[TODO: optional suggested reviewers, if the submission system asks for them; avoid conflicts of interest.]

Sincerely,

[Corresponding author name, affiliation, e-mail, ORCID iD]
on behalf of all authors
