# Cover letter (draft)

*SAUCIS does not require a cover letter in its author guidelines; the DergiPark submission form may offer a field for one. SAUCIS does not ask for highlights, so none are provided. Fill the bracketed fields; do not add result claims before the experiments are complete.*

---

[Date]

To the Editor-in-Chief
Sakarya University Journal of Computer and Information Sciences (SAUCIS)

Dear Editor,

We submit the manuscript "Size-Agnostic Safe Reinforcement Learning for V2G Charging with a Feasibility-Preserving Safety Layer" [TODO: final title, chosen by the authors] for consideration as a research article in SAUCIS.

Charge point operators that offer vehicle-to-grid services must respect a transformer limit and deliver the energy each driver requests by departure. Reinforcement learning controllers for this task usually encode both requirements as reward penalties, and their networks are tied to one station size. The manuscript combines size-agnostic set policies (attention and Deep Sets), in which every charger is a token and one set of weights serves any number of chargers, with a safety layer that projects the raw actions onto taper-aware per-vehicle bounds and the transformer limit. The main theorem has two layers. Under stated simulator and grid-headroom conditions, the transformer limit holds at every step for every sequence of raw actions. The departure requirement is met for every vehicle whose stay contains no infeasible step, and the paper measures how often learned and rule-based controllers cause such steps. The paper states openly that the projection has a known algebraic form and that each ingredient exists in some form, and it places its contribution in the combination, the two-layer result and its measured cost.

The evaluation trains on stations with 10 and 20 chargers in the open EV2Gym simulator and tests without retraining on stations with up to 200 chargers and on a tighter transformer rating. All methods receive the same admitted requirements. The main metric is the gap of a penalised profit to a perfect-information optimum that bounds every compared method, and a penalty-trained twin measures the cost of the guarantee. Baselines include a penalty-trained twin, a Lagrangian constrained-RL method, rule-based schedulers and a receding-horizon model-predictive controller, each also with the safety layer. With two training seeds per learned method, no transformer overload occurred in any episode run through the safety layer. The paper also reports its negative findings openly. The attention policy showed no advantage over Deep Sets, and a price-threshold rule with the same layer had a lower mean gap than the attention policy at every station size of the test grid. With the departure bound, the attention policy left more vehicles unserved than an attention variant without the bound and than its penalty-trained twin, so the conditional departure guarantee is most useful with controllers that do not ride their lower bounds. [TODO: the authors confirm or rewrite this summary.]

The topic lies at the intersection of machine learning, optimisation and smart-grid applications. [TODO: optionally cite one or two related SAUCIS articles that the authors have checked.] We believe the paper fits the journal's scope of artificial intelligence and its applications.

The manuscript is original, has not been published before and is not under consideration elsewhere. All authors approved the submission. The code, scenario configurations and analysis scripts will be made public [TODO: repository link or "on acceptance"]. The use of an AI assistant (Claude, Anthropic) for literature work, method derivation, experiment design, code and text drafting is disclosed in full in the Artificial Intelligence Statement of the Article Information Form. [TODO: the authors confirm how the experiments were run and who wrote the interpretation.]

[TODO: optional suggested reviewers, if the submission system asks for them; avoid conflicts of interest.]

Sincerely,

[Corresponding author name, affiliation, e-mail, ORCID iD]
on behalf of all authors
