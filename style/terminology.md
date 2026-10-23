# Terminology (one name per concept; A5 must use these exact terms)

| Concept | Use this term | Do NOT use |
|---|---|---|
| The closed-form action-correction layer | **safety layer** (first use: "safety layer, a closed-form projection onto the feasible set") | projection layer, shield, safety filter, action correction (except when describing other papers) |
| The mathematical operation inside it | **projection** | clipping layer, correction |
| The proposed policy network | **attention policy** | transformer policy, token policy |
| Full method (policy + safety layer) | **SA-Attn** (placeholder name, to be fixed at G2) | mixed names |
| Unit of the station | **charger** (a charger may host one EV) | charging point, port, EVSE, socket (except quoting EV2Gym) |
| Station operator | **charge point operator (CPO)** | aggregator (unless referring to other work) |
| Grid-side limit | **transformer limit** | transformer constraint and grid constraint alternately |
| EV-side requirement | **departure energy requirement** | user constraint, user satisfaction constraint (except for EV2Gym's metric "user satisfaction") |
| Transfer without retraining | **zero-shot transfer** | generalisation, scale transfer alternately |
| Gap to the perfect-information optimum | **optimality gap** (w.r.t. the **perfect-information optimum**) | oracle gap, regret |
| Step where Σ l_i > C(t) | **infeasible step**; handled by the **fallback rule** | violation step |
| Arrival rejection | **admission control** | |
| Discharging to the grid | **V2G discharging** | reverse charging |
| Profit difference penalty-based vs safety layer | **cost of the guarantee** | price of safety |

| Family of size-agnostic architectures (attention and Deep Sets) | **set policy** (allowed since DD45) | |
