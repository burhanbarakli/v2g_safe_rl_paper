# Cover letter (draft)

*SAUCIS does not require a cover letter in its author guidelines; the DergiPark submission form may offer a field for one. SAUCIS does not ask for highlights, so none are provided. Fill the bracketed fields; do not add result claims before the experiments are complete.*

---

[Date]

To the Editor-in-Chief
Sakarya University Journal of Computer and Information Sciences (SAUCIS)

Dear Editor,

We submit the manuscript "Attention-Based Safe Reinforcement Learning for Scale-Agnostic and Constraint-Guaranteed V2G Charging Control" for consideration as a research article in SAUCIS.

Charge point operators that offer vehicle-to-grid services must respect a transformer limit and deliver the energy each driver requests by departure. Reinforcement learning controllers for this task usually encode both requirements as reward penalties, and their networks are tied to one station size. The manuscript combines an attention policy, in which every charger is a token and one set of weights serves any number of chargers, with a closed-form safety layer. Under stated assumptions, we prove that the transformer limit and every departure energy requirement hold at every step for any action of the policy, during training and deployment. The paper states openly that the projection inside the safety layer has a known algebraic form, and it places its contribution in the bounds, the multi-step proof and the evaluation.

The evaluation trains on stations with 10 and 20 chargers in the open EV2Gym simulator and tests without retraining on stations with up to 200 chargers. It reports the optimality gap to a perfect-information linear program with hard constraints and the profit given up for the guarantee. [RESULT: one sentence with the main finding, to be added by the authors after the experiments.]

The topic lies at the intersection of machine learning, optimisation and smart-grid applications, and SAUCIS has published work on reinforcement learning and on electric-vehicle energy forecasting. We believe the paper fits the journal's scope of artificial intelligence and its applications.

The manuscript is original, has not been published before and is not under consideration elsewhere. All authors approved the submission. The code, scenario configurations and analysis scripts will be made public [TODO: repository link or "on acceptance"]. The use of an AI assistant in the preparation of the manuscript is disclosed in the Artificial Intelligence Statement of the Article Information Form.

[TODO: optional suggested reviewers, if the submission system asks for them; avoid conflicts of interest.]

Sincerely,

[Corresponding author name, affiliation, e-mail, ORCID iD]
on behalf of all authors
