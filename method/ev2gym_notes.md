# EV2Gym technical notes (for the safe-RL V2G paper)

Agent A2 (Method and Theory), task 1. Written 2026-10-21.

- Source: `/home/claude/work/ext/EV2Gym`, commit `55ef368c25a58476a9eeaed5111f112a63a38ca8` (checked with `git log -1`; `git status` shows no tracked file changed, only ignored `__pycache__/` and `ev2gym.egg-info/` from `pip install -e`).
- Package: `ev2gym` 2.0.0 (`setup.py:12`), MIT licence. Paper: Orfanoudakis et al., IEEE T-ITS 2024, doi 10.1109/TITS.2024.3510945 (`README.md:15,342-353`).
- Python and venv: Python 3.13.16. Venv at `/home/claude/work/ext/venv` with numpy 2.4.6, pandas 2.3.3, gymnasium 1.4.0, gurobipy 13.0.3, pandapower 3.5.6, numba 0.68.0, scipy 1.18.1, multicopula 0.0.2, psutil 7.2.2, setuptools 80.10.2, matplotlib 3.11.2, networkx 3.7, PyYAML 6.0.3.
- Install caveat: `setup.py:30-37` lists only gymnasium, pyyaml, matplotlib, pandas, networkx and gurobipy. Importing the env also needs `pandapower`, `numba`, `scipy`, `psutil`, `multicopula` (imported unconditionally via `loaders.py:17 -> grid.py:12-19 -> data_augment.py:3`) and `setuptools<81` (for `pkg_resources`, `loaders.py:9`). `requirements.txt` pins numpy 1.24 and pandas 2.0, but numpy 2.4 and pandas 2.3 work for the env itself (verified).
- Gurobi: `gurobipy` 13.0.3 installs with a **restricted, size-limited licence** ("Restricted license - for non-production use only - expires 2027-11-29"). A 2001-variable LP fails with "Model too large for size-limited license" (limit 2000 variables/constraints). There is no full Gurobi licence here.
- All file:line references below are relative to `ev2gym/` unless they start with the repo root. Test scripts are in `/tmp/claude-0/-home-claude/95c38eef-af15-5fc2-9c05-762613643456/scratchpad/t1.py ... t6.py`. They must run from the repo root because config paths such as `ev_specs_file: ./ev2gym/data/...` are relative.

---

## 1. Time step, horizon, dates

- **Step length**: config key `timescale` in minutes (`ev2gym_env.py:125`). It is 15 in all example configs (`example_config_files/V2GProfitMax.yaml:6`). The date advances by `timedelta(minutes=timescale)` per step (`ev2gym_env.py:558-561`). Steps per day = 1440/timescale (96 at 15 min, 288 at 5 min).
- **Horizon**: `simulation_length` in steps (`ev2gym_env.py:78,127`). V2GProfitMax.yaml has 112, i.e. 28 h starting at 05:00 (`V2GProfitMax.yaml:7,22-23`). V2Ggrid.yaml and simplePST.yaml have 96. The episode ends when `current_step >= simulation_length` (`ev2gym_env.py:460`). `truncated` is always False (`ev2gym_env.py:451`).
- **Start date**:
  - If `random_day: True` (`V2GProfitMax.yaml:17`), the date is 2022-01-01 at `hour:minute` plus `randint(0, 547)` days, so 2022-01-01 to about 2023-07-01. It is re-drawn at every `reset()` (`ev2gym_env.py:130-154, 262-290`).
  - `random_hour: True` draws the hour from 5-15 (`ev2gym_env.py:131-133`).
  - `simulation_days: weekdays|weekends|both` moves the date forward (`ev2gym_env.py:147-154`). The `workplace` scenario always skips weekends (`ev2gym_env.py:142-145`).
  - If `random_day: False`, `year/month/day/hour/minute` are used (`ev2gym_env.py:155-161`) and are restored on every reset (`ev2gym_env.py:262-263`).
- **dt can be changed**: verified with timescale=5 and simulation_length=288. The env ran, prices had shape (25, 288), and the episode covered 24 h.
  - Prices are hourly and held constant within the hour (`loaders.py:434-461`). Any timescale works for prices.
  - Residential loads (15-min data) need `15 % timescale == 0` when timescale < 15, because rows are repeated `15/timescale` times (`loaders.py:126-133`). Use 5, 15, 30 or 60, not 10.
  - PV (hourly data) needs `60 % timescale == 0` (`loaders.py:184-199`).
  - Arrival rates scale with `timescale/60` (`utils.py:538`). Stay durations are converted with `60/timescale` (`utils.py:240`). `min_time_of_stay // timescale` gives the minimum stay in steps (`utils.py:497-498`).

## 2. Action space and action-to-power mapping

- **Space**: `Box(low=-1 or 0, high=1, shape=(number_of_ports,), float64)`. There is one entry per **port**, ordered charger by charger and then port by port. The low bound is -1 when `v2g_enabled` (`ev2gym_env.py:225-231`). `number_of_ports = Σ cs.n_ports` (`ev2gym_env.py:201-202`). V2GProfitMax.yaml has 25 chargers × 1 port, so the shape is (25,) (verified).
- **Per-charger processing** in `EV_Charger.step` (`models/ev_charger.py:114-233`):
  1. **Empty ports**: the action is overwritten with 0 and `invalid_action_punishment += 1` (`ev_charger.py:137-140`). This writes into the caller's numpy array, because `actions[port_counter:port_counter+n_ports]` is a view (`ev2gym_env.py:365-366`).
  2. **Shared charger current (multi-port only)**: if Σ actions over the charger's ports > 1 (or < -1), all port actions are divided by |Σ| (`ev_charger.py:143-149`). This does nothing with 1 port per charger.
  3. Each action is rounded with `round(action, 5)` (`ev_charger.py:157`). The range check `assert (cond, msg)` is a tuple and so always passes (`ev_charger.py:158-159`), meaning **out-of-range actions are not rejected**. The current is still capped later by the EV power limits.
  4. **Charge** (a>0): `amps = a * max_charge_current`. It is set to 0 if below `min_charge_current - 0.01` (`ev_charger.py:167-170`).
  5. **Discharge** (a<0): `amps = a * |max_discharge_current|`. If `amps > min_discharge_current - 0.01` it is set to `min_discharge_current` (`ev_charger.py:183-186`).
  6. `ev.step(amps, voltage, phases=cs.phases)` is called. Profit is `|ΔE|*charge_price` for charging or `|ΔE|*discharge_price` for discharging (`ev_charger.py:178,194`). The charger power is `cs.current_power_output += ΔE * 60/timescale` in kW (`ev_charger.py:180,196`).
  7. An Exception is raised if the summed actual amps exceed `max_charge_current` (`ev_charger.py:203-205`).
- **Electrical formula** (`ev.py:169, 279, 365`; charger power helpers `ev_charger.py:251-258`):
  - φ = min(cs.phases, ev.ev_phases). V_eff = V·√φ with V = `charging_station.voltage` (400 V, labelled "Line-Line" in `V2GProfitMax.yaml:128`).
  - Commanded power P_cmd [kW] = I[A]·V·√φ / 1000 = a · I_max · V·√φ / 1000.
  - With the defaults (I_max = 32 A, V = 400 V, φ = 3), **P_cs,max = 22.17 kW** in both directions (`cs.get_max_power()` = 22.170 kW and `cs.get_min_power()` = -22.170 kW, verified).
  - The EVs cap power much lower: `max_ac_charge_power` / `max_ac_discharge_power` are 11 kW (7.4 kW for the Peugeot 208) in `data/ev_specs_v2g_enabled2024.json`. So **a ≈ 0.496 already saturates an 11 kW EV.**
- **Clipping to EV limits happens in ev.py, not ev_charger.py**:
  - Charge: battery-side ΔSoC is capped at η·P_ev,max·dt/C (`ev.py:297-301`).
  - Discharge: |P_cmd| is clipped to |max_discharge_power| (`ev.py:370-371`).
  - EV minimum-power dead-bands: `ev.py:151-154`. Both are 0 by default.
- **Current/power limits from config** (`V2GProfitMax.yaml:123-129`): `min_charge_current: 0`, `max_charge_current: 32`, `min_discharge_current: 0`, `max_discharge_current: -32`, `voltage: 400`, `phases: 3`.
  - If `v2g_enabled: False`, the discharge currents are forced to 0 (`loaders.py:346-351`).
  - A topology JSON (`example_config_files/charging_topology_two_areas.json`) can set these per charger (`loaders.py:314-343`).
- **Built-in action wrappers** (`rl_agent/action_wrappers.py`):
  - `BinaryAction` (l.8), `ThreeStep_Action` (l.50), `ThreeStep_Action_DiscreteActionSpace` (l.93) and `mask_fn` (l.141).
  - `Rescale_RepairLayer` (l.159-451) is a heuristic repair for **power-setpoint tracking** (G2V only, 1 port per charger). It is not a transformer or departure-energy safety layer.
  - `MinMax_RepairLayer` raises NotImplementedError (l.463).
  - `rl_agent/noise_wrappers.py` has `FailedActionCommunication` (l.12) and `DelayedObservation` (l.62).

## 3. EV model (`models/ev.py`)

- **Attributes** (`ev.py:45-113`):
  - `time_of_arrival` and `time_of_departure` are in steps.
  - `battery_capacity_at_arrival`, `battery_capacity`, `desired_capacity` (defaults to `battery_capacity`), `min_battery_capacity` and `min_emergency_battery_capacity` are in kWh.
  - Power limits: `max_ac_charge_power`, `min_ac_charge_power`, `max_discharge_power` (negative) and `min_discharge_power`.
  - `ev_phases`, `transition_soc` and `transition_soc_multiplier`.
  - `charge_efficiency` and `discharge_efficiency` are scalars or dicts {current A → %}.
  - State: `current_capacity` (kWh), `current_energy` (kWh in the last step, despite the "kW" comment), `required_energy` (= capacity − energy, i.e. energy-to-full and **not** to desired; `ev.py:101,353-354,399`).
  - There is no V2G flag per EV. V2G is enabled per environment (`v2g_enabled`), and an EV with `max_ac_discharge_power = 0` (ev_specs.json) simply cannot discharge.
- **Charging update** `_charge` (`ev.py:240-355`). This is the ACN-Sim two-stage model, done in SoC units:
  - η = `charge_efficiency`, or `charge_efficiency.get(round(amps),1)/100` if it is a dict (`ev.py:287-290`). **η depends on the commanded current, rounded to an integer amp.**
  - pilot_dsoc = η·P_cmd·(dt/60 h)/C and max_dsoc = η·P_ev,max·dt/C. pilot_dsoc is capped at max_dsoc (`ev.py:295-301`).
  - If `transition_soc == 1`: new SoC = min(1, SoC + pilot_dsoc) (`ev.py:303-306`). This is **linear** up to the full-battery cap.
  - Otherwise (CC-CV taper, `ev.py:312-344`):
    - pilot_transition_soc = ts + (pilot_dsoc − max_dsoc)/max_dsoc·(ts−1), so a lower pilot moves the transition point up.
    - Above it: new SoC = 1 + exp(m·pilot_dsoc/(pts−1))·(SoC−1), with m = `transition_soc_multiplier`.
    - Across the transition there is a mixed formula. The result is always capped at SoC + min(pilot_dsoc, max_dsoc).
  - ΔE = ΔSoC·C. **Reported power = ΔE/dt = η·min(P_cmd, P_ev,max) in the CC region** (`ev.py:346-355`).
- **Discharging update** `_discharge` (`ev.py:357-405`):
  - given_power = P_cmd (negative), clipped to `max_discharge_power` (`ev.py:367-371`).
  - η_d = `discharge_efficiency` or the dict lookup with |round(amps)| (`ev.py:375-379`).
  - ΔE = given_power·η_d·dt (`ev.py:381`). The floor is `min_battery_capacity`: if crossed, ΔE = −(E − E_min), or 0 if already at the floor (`ev.py:382-397`).
  - The emergency metric is counted when crossing `min_emergency_battery_capacity`, which is monitored and not enforced (`ev.py:401-402`; `V2GProfitMax.yaml:148`).
- **Efficiency handling is not physical**. η scales the *battery-side* energy in both directions, and that same battery-side ΔE is what the charger reports as power, bills (profit) and sends to the transformer (`ev_charger.py:178-181,194-197`; `ev2gym_env.py:375-379`). As a result:
  - charging: grid-reported energy = battery gain = η·P_cmd·dt;
  - discharging: grid-reported energy = battery loss = η_d·P_cmd·dt.
  - So **energy accounting is lossless**: Σ billed energy = ΔE_battery. η only derates the deliverable power for a given current.
  - Verified (`t3.py`):
    - η_c=0.9 and 16 A (11.09 kW commanded, capped at 11 kW) give reported 9.9 kW and the battery gains exactly 2.475 kWh in 15 min.
    - η_d=0.8 and −16 A give reported −8.8 kW and the battery loses exactly 2.2 kWh.
  - There are separate η_c and η_d only when they are scalars:
    - ev_specs files without efficiency tables: η_c, η_d ~ U[0.95, 1] drawn independently (`utils.py:292-296`).
    - Homogeneous specs: config `ev.charge_efficiency` / `ev.discharge_efficiency` (`utils.py:342-343`), which are 1 in V2GProfitMax.yaml.
    - With `ev_specs_v2g_enabled2024.json` (used by V2GProfitMax.yaml), the discharge dict is a **copy** of the 3-phase charge-efficiency table (`utils.py:273-290`). Values are 84-93 % depending on the current level, and missing or zero levels are filled from the nearest non-zero level.
  - Edge case: `dict.get(round(amps), 1)/100` would give η = 0.01 for currents above 100 A (`ev.py:288`). This does not occur at 32 A.
- **Rounding**: after every non-zero step, `current_capacity = ceil(current_capacity, 0.01 kWh)` (`ev.py:182-183,188-189`). The reported ΔE is computed **before** rounding, so the battery can gain up to +0.01 kWh per active step that is not reported. Verified: 0.0173 kWh reported, capacity 20.00 → 20.02.
- **SoC limits**:
  - Max SoC is 1 (capacity `battery_capacity`, `ev.py:305,326-344`).
  - Min is `min_battery_capacity` in kWh: config `ev.min_battery_capacity` (5 kWh in V2GProfitMax.yaml:146, passed at `utils.py:313,335`).
  - At arrival, the energy is raised to `min_battery_capacity` if lower (`utils.py:228-229`).
- **Desired energy at departure**:
  - `desired_capacity = config ev.desired_capacity (fraction) × battery_capacity` (`utils.py:314-315,327-328`). It is **1.0, i.e. a full battery**, in all example configs (`V2GProfitMax.yaml:149`).
  - Generation (`utils.py:200-229`): required energy ~ N(μ(arrival half-hour, scenario), 0.5μ) from `data/mean-demand-per-arrival.csv`, redrawn U{5..9} if < 5 kWh. Then E_arr = C − E_required. If E_arr > desired_frac·C, E_arr is redrawn as `randint(1, C)` (so with desired_frac < 1 some EVs arrive already above target). E_arr is floored at `min_battery_capacity`.
  - Energy needed = `desired_capacity − battery_capacity_at_arrival`. It is 7-30 kWh in a sample day (seed 5).
- **Arrival and departure**:
  - All EVs of an episode are pre-generated at `reset()` in `env.EVs_profiles` (`ev2gym_env.py:294`; `utils.py:477-557`). For each port and step t ∈ [2, T − min_stay_steps − 1), an arrival occurs if the port has been free for 3 steps and `U·100 < rate(scenario, 15-min slot)·(timescale/60)·spawn_multiplier` (`utils.py:504-553`). Rate tables: `data/distribution-of-arrival.csv` (weekday) and `-weekend.csv`. The workplace scenario has arrivals only 06-18 h on weekdays (`utils.py:512-521`).
  - Stay = N(μ(arrival slot), 0.2μ) hours from `data/mean-session-length-per.csv`, converted to steps `·60/timescale + 1`, and floored at `min_time_of_stay // timescale` (`utils.py:231-252`).
  - `time_of_arrival = t+1` and `time_of_departure = int(stay + t + 3)` (`utils.py:316-318`).
  - With `empty_ports_at_end_of_simulation=True` (constructor default, `ev2gym_env.py:52`), EVs whose departure would fall within 4 steps of the episode end are **not spawned** (`utils.py:254-256`). So every EV departs inside the episode.
  - **The departure time is deterministic and known at arrival.** It is stored in `ev.time_of_departure`. The "probabilistic departure" in the docstring (`ev.py:22`) is not implemented.
- **Timing (verified, `t2.py`)**:
  - An EV is inserted at the end of step `time_of_arrival − 1` (`ev2gym_env.py:399-417`), so the first action that reaches it is at step k = t_arr.
  - At step k = t_dep the charging action is applied first, and then `is_departing(cs.current_step)` removes it (`ev_charger.py:209-229`, `ev.py:191-202`).
  - **The EV is controlled for steps k = t_arr ... t_dep inclusive, i.e. t_dep − t_arr + 1 actions.** `max_energy_AFAP` uses the same range (`ev.py:435`).
  - The observation is built after `current_step` is incremented, so at decision time k the number of remaining actions **including the current one is τ = t_dep − k + 1**. The `V2G_profit_max` state feature `time_of_departure − current_step` equals τ − 1.
  - EVs cannot stay after t_dep. A new EV can take the port only from step t_dep + 3, so the port is empty for at least 2 steps (`utils.py:534-536,551-552`; measured minimum t_arr,next − t_dep,prev = 3 over 30 seeds).
- **Charge power curve**: the nonlinear taper above is the only SoC-dependent max power (verified, `t3.py`).
  - With `transition_soc=0.8` and multiplier 5, full power (11 kW) holds up to SoC ≈ 0.93. Then 10.4 → 2.6 → 0.63 → 0.15 kW per 15-min step.
  - With multiplier 1 the taper starts at 0.8 (10.9, 8.9, 6.8, 5.1, ... kW).
  - Under the taper, a pilot below the SoC-dependent limit is still delivered linearly (at SoC 0.85, P = P_cmd for P_cmd ≤ 8.3 kW).
  - **Heterogeneous specs ignore config `ev.transition_soc` and draw transition_soc ~ U[0.7, 0.9] for every EV** (`utils.py:309-310`). The multiplier comes from config (`transition_soc_multiplier: 5`, `V2GProfitMax.yaml:151`). Homogeneous specs use `ev.transition_soc` (1 in V2GProfitMax.yaml, i.e. no taper).
- **Heterogeneous specs**: if `heterogeneous_ev_specs`, the model is sampled by registrations from `ev_specs_file` (`loaders.py:26-42`; `utils.py:213-216`).
  - `ev_specs_v2g_enabled2024.json` has 8 models: Tesla M3/MY 57.5 kWh, Kia Niro 64.8, ID.3 58, Enyaq 58, Kona 64, Peugeot 208 46.3 (7.4 kW), ID.4 77.
  - All are 11 kW AC except the Peugeot, and all are V2G-capable with the same AC discharge limit.
  - `ev_phases` is hard-coded to 3 (`utils.py:319`).

## 4. Charger model (`models/ev_charger.py`)

- Constructor defaults (`ev_charger.py:41-55`): I_ch ∈ [0, 56] A, I_dis ∈ [−56, 0] A, 230 V, n_ports=2, AC, 3 phases. The config overrides them (`loaders.py:353-366`): 0/32/0/−32 A, 400 V, 3 phases, `n_ports = number_of_ports_per_cs` (`V2GProfitMax.yaml:44`, set to 1).
- There is **no charger efficiency**. All efficiency is in the EV.
- `charger_type="DC"` raises NotImplementedError in `ev.step` (`ev.py:148-149`).
- Multiple EVs per charger: one EV per port. Ports share the charger current through the |Σa| ≤ 1 normalisation (`ev_charger.py:143-149`), and the summed current must stay ≤ `max_charge_current` (`ev_charger.py:203-205`). `spawn_ev` takes the first free port (`ev_charger.py:266-286`).
- The MPC baselines and `Rescale_RepairLayer` require one port per charger (`baselines/mpc/mpc.py:58`; `action_wrappers.py:186-187`).

## 5. Transformer model (`models/transformer.py`)

- `max_power` in kW comes from config `transformer.max_power` (100 kW, `V2GProfitMax.yaml:119-120`, used at `loaders.py:290`) or from the topology JSON per transformer (`loaders.py:271`). It is stored as a per-step array `max_power[t]`, with `min_power = −max_power` (`transformer.py:43-46`). Currents use V·√phases (`transformer.py:39-41`).
- **Number of transformers and chargers per transformer**:
  - `number_of_transformers` (`V2GProfitMax.yaml:43`). Chargers are assigned round-robin as `cs_transformers = [0..n_tr−1]*(n_cs//n_tr) + ...` (`loaders.py:509-513`).
  - A topology JSON gives explicit transformer → charger lists (`loaders.py:260-278, 314-343`; `ev2gym_env.py:177-186`).
  - With `simulate_grid: True` the grid file sets the transformers (`loaders.py:475-507`).
- **Power bookkeeping per step**:
  1. `tr.reset(step)` sets `current_power = inflexible_load[k] + solar_power[k]` (solar is negative) (`transformer.py:258-267`; called at `ev2gym_env.py:359-360`).
  2. Each charger adds its reported power: `tr.step(amps, cs.current_power_output)` (`ev2gym_env.py:378-379`; `transformer.py:269-274`).
- **Overload**:
  - `is_overloaded`: P > max_power[k] + 1e-4 or P < min_power[k] − 1e-4 (`transformer.py:276-290`).
  - `get_how_overloaded` = |P − max_power[k]| in **kW** (`transformer.py:292-302`). For reverse flow below −max this formula overstates the overload (it measures the distance to +max).
  - Logged into `env.tr_overload[tr, k]` (`ev2gym_env.py:526-527`). The statistic `total_transformer_overload = Σ_k tr_overload`, a kW sum that is **not** multiplied by dt despite the "kWh" label (`utils.py:26,172`).
  - **Power is never clipped or curtailed.** Verified with max_power=30 kW and charge-as-fast-as-possible: station power reached 113 kW, 15 overloaded steps, overload sum 649. The episode continues and the reward is only penalised if the reward function does so.
- **Inflexible load** (`inflexible_loads.include`, `V2GProfitMax.yaml:78-82`):
  - Residential 15-min data `data/residential_loads.csv`: 25 houses, the sum of 10 random columns per transformer (`loaders.py:107-161`). It is used for **all** scenarios (TODO at `loaders.py:246-248`).
  - Scaled so that the max over the episode = N(mult, 0.1)·P_tr. With `inflexible_loads_capacity_multiplier_mean: 1` it peaks at about 100 % of transformer rating, then is clipped to ±max (`transformer.py:215-236`).
  - Forecast = N(forecast_mean %·load, forecast_std %·load), clipped (`transformer.py:238-256`). With `forecast_mean: 30` the forecast mean is 30 % of the true load, a strongly biased forecast. These semantics look odd.
- **PV** (`solar_power.include`): `data/pv_netherlands.csv` (hourly, normalised, max 0.853), × U(0.9, 1.1) per transformer (`loaders.py:164-225`). Then `solar_power = −pv·N(mult,0.1)·P_tr` (`transformer.py:190-198`). The forecast is built the same way (`transformer.py:200-213`).
- **Demand response** (`demand_response.include`, `V2GProfitMax.yaml:92-103`): `events_per_day` events with random start N(12 h, 2 h), length 60-60 min, and capacity reduction N(35 %, 5 %) of max_power (`transformer.py:80-140`). The event is announced `notification_of_event_minutes` ahead through `get_power_limits(step, horizon)` (`transformer.py:142-171`).
- V2GProfitPlusLoads.yaml is V2GProfitMax.yaml with all three switched on (diff lines 79, 86, 93). V2GProfitMax.yaml has all three off, so there the only limit is the constant 100 kW.
- Randomness of loads, PV and DR uses `tr_rng` seeded by `tr_seed` (−1 means same as the env seed, `ev2gym_env.py:97-100,255-257`). The load and PV forecast noise uses the global `np.random`.

## 6. Prices

- `load_electricity_prices` (`utils/loaders.py:395-464`) reads `data/Netherlands_day-ahead-2015-2024.csv` (hourly NL day-ahead in €/MWh, 2015 to 2025-07-01).
- For each step it takes the price of the step's (year, month, day, **UTC hour**). The sim date is naive and is matched against the `Datetime (UTC)` column (`loaders.py:417-445`). If no match, the 2022 price for the same date is used (`loaders.py:446-457`).
- Arrays have shape (n_cs, T) and are identical for all chargers:
  - `charge_prices = −price/1000` (€/kWh, **negative**, `loaders.py:442-443`);
  - `discharge_prices = +price/1000 × discharge_price_factor` (`loaders.py:444-445,463`; factor 1 in `V2GProfitMax.yaml:35`).
- **Sign convention**: per-step "costs" (really profit) are Σ|ΔE_ch|·charge_price + Σ|ΔE_dis|·discharge_price (`ev_charger.py:178,194`). So profit = Σ_t λ_t (E_dis − E_ch) in €, with ΔE the battery-side energy (equal to the reported energy, see §3). Negative prices are handled correctly (charging then earns money). `total_profits` = Σ over chargers (`utils.py:15-16`).
- Price statistics over the dates that `random_day` can draw (2022 to 2023-H1): mean 181, median 148, std 126, min −500, max 871 €/MWh; 2.1 % of hours are negative.
- **Forecasts in the state**: `V2G_profit_max` and `V2G_profit_max_loads` include |charge_prices| for the next 20 steps, **perfect foresight**, zero-padded (`rl_agent/state.py:76-82,119-125`). The MPCs use the full true price vector (`baselines/mpc/mpc.py:200-207`).

## 7. Reward, cost and state functions

All are in `rl_agent/reward.py`. `total_costs` is the per-step profit (€) from §6, `sat` is the departing EV's satisfaction E/E_des capped at 1 (`ev.py:204-214`), and k = current_step − 1.

| Function | Formula |
|---|---|
| `SquaredTrackingErrorReward` (l.7) | −(min(PS_k, PP_k) − P_k)², where PS is the power setpoint and PP the charge-power potential (`utils.py:760-791`) |
| `SqTrError_TrPenalty_UserIncentives` (l.16) | −(min(PS, PP, Ptr_max) − P)² − 100·Σ overload − 1000·Σ(1 − sat) |
| `ProfitMax_TrPenalty_UserIncentives` (l.34) | profit − 100·Σ_tr overload_kW − 100·Σ_dep exp(−10·sat) |
| `SquaredTrackingErrorRewardWithPenalty` (l.46) | as above, with −100 when P=0 and PP>0 |
| `SimpleReward` (l.60) | −(PS − P)² |
| `MinimizeTrackerSurplusWithChargeRewards` (l.67) | −(P − PS)²·1[P>PS] + P |
| `profit_maximization` (l.78) | profit − 100·Σ_dep exp(−10·sat). Used for V2GProfitMax.yaml in `evaluator.py:74-76` |
| `V2G_grid_full_reward` (l.89) | profit − Σ_dep (E − E_des)² + 1000·Σ_nodes min(0, 0.05 − abs(1 − v)) |
| `V2G_grid_simple_reward` (l.114) | 1000·voltage loss only |
| `V2G_profitmax` (l.123) | profit − 100·Σ_dep max(0, E_des − E) |
| `V2G_costs_simple` (l.150) | profit |
| `V2G_profitmaxV2` (l.155) | profit − 0.05·Σ_conn (E_min,k − E)²·1[infeasible-trend] − 0.05·Σ_dep (E_des − E)²₊. Here E_min,k = E_des − (τ_dep+1)·P_ev,max·dt, an online "latest-feasible" shaping term close to our l_i |
| `Grid_V2G_profitmaxV2` (l.215) | V2G_profitmaxV2 + 50 000·voltage loss |
| `pst_V2G_profitmaxV2` (l.281) | V2G_profitmaxV2 + 1000·min(0, PS − P) |

- **Costs** (`rl_agent/cost.py`, for constrained RL, via the `cost_function` argument in `ev2gym_env.py:434-438`):
  - `transformer_overload_usrpenalty_cost` (l.8) = 100·Σ overload + 100·Σ exp(−10·sat).
  - `ProfitMax_TrPenalty_UserIncentives_safety` (l.22) = profit, as a reward.
- **States** (`rl_agent/state.py`):
  - `PublicPST` (l.6): [k/T, PS_k, P_{k−1}] + per port [full?1:0.5, total energy exchanged, k − t_arr]. This is the env default (`ev2gym_env.py:46`).
  - `V2G_profit_max` (l.65): [k, P_{k−1}, 20 future |prices|] + per port [SoC, t_dep − k], zeros if the port is empty. Dimension 22 + 2N (72 for N=25, verified).
  - `V2G_profit_max_loads` (l.108): as above + per transformer [20-step (load − PV) forecast, 20-step power limits].
  - `BusinessPSTwithMoreKnowledge` (l.159) is broken (`EV.etime_of_departure` does not exist).
  - `V2G_grid_state` (l.216): weekday, hour sin/cos, price, PS, P, node P/Q, per port [E, t_dep − k + 1, bus].
  - **No state function includes E_des or E_req per EV.** Departure energy is implicit (full battery). We need our own state function or wrapper.

## 8. Built-in baselines

**Heuristics** (`baselines/heuristics.py`). All return actions in [0,1] (G2V only):
- `RoundRobin` (l.7-95): power-setpoint based.
- `ChargeAsLateAsPossible` (l.98-149): full action when t_dep − ceil((1−SoC)/(P_max·dt/C)) ≤ k. It ignores η and the taper.
- `ChargeAsFastAsPossible` (l.152-166): all ones, including empty ports.
- `ChargeAsFastAsPossibleWithPowerLimit(env, power_limit)` (l.168-229): random subset of EVs.
- `ChargeAsFastAsPossibleToDesiredCapacity` (l.231-268).
- `RoundRobin_GF`, `RoundRobin_GF_off_allowed` (l.270, 402).
- `DoNothing` (l.533) and `RandomAgent` (l.546).
- `ChargeAsLateAsPossibleToDesiredCapacity` (l.561-622).
- None of them respects the transformer limit, except setpoint-based RoundRobin when the setpoint is the limit.

**MPC** (`baselines/mpc/`). All use Gurobi, all need 1 port per charger, and all **read the full future EV list `env.EVs_profiles` and true prices** (perfect arrival and departure information inside the horizon) (`mpc.py:36,108-159,200-207`).
- Decision variables are grid-side power u_ch, u_dis ≥ 0 per port and step. There is a binary Zbin to forbid simultaneous charge and discharge.
- Battery model: x_{t+1} = x_t + η_c·T·u_ch − η_d·T·u_dis (`mpc.py:323-340`). This matches EV2Gym's (non-physical) convention. η is taken from the **first** EV's dict maximum (`mpc.py:110-119`).
- **Departure energy is a HARD constraint**: XF = desired capacity at the departure step and XF = min_SoC·C otherwise, enforced as x ≥ XF (`mpc.py:307-321, 375-378`). The MPC departure index is t_dep + 1 (`mpc.py:128-131`).
- Classes:
  - `eMPC_V2G` / `eMPC_G2V` (`eMPC.py:16, 166`): economic MPC with the transformer constraint Σ(u_ch − u_dis) + L + PV ≤ P_tr(forecast) and ≥ −max (`eMPC.py:100-121`). It returns all-zero actions if infeasible (`eMPC.py:133-137`).
  - `eMPC_V2G_v2` / `eMPC_G2V_v2` (`eMPC_v2.py:16, 261`).
  - `OCMF_V2G` / `OCMF_G2V` (`ocmf_mpc.py:16, 186`): the Optimal Charging with Max Flexibility variant, which reserves capacity CapF1.
  - `V2GProfitMaxOracle` (`V2GProfitMax.py:16-169`): one MILP over the whole episode, **without** a transformer constraint (commented out, l.112-122).
  - `V2GProfitMaxLoadsOracle` (`V2GProfitMax.py:172-325`): with the transformer constraint against true loads, PV and limits (`mpc.py:256-264`).
- Status in our venv:
  - `eMPC_V2G` with horizon 10 runs for N=2 and N=25 (2·25·10 + 25·10 = 750 variables, under the 2000 cap).
  - **`V2GProfitMaxOracle` crashes under numpy 2** (`TypeError ... only 0-dimensional arrays` at `V2GProfitMax.py:126`, `obj_expr.addTerms(f[i], u[i])` with f of shape (n,1)). In any case it exceeds the restricted licence for N=25 and T=112 (8400 variables).

**Gurobi perfect-information models** (`baselines/gurobi_models/`). They are built from a **replay pickle**: run the env with `save_replay=True` (`ev2gym_env.py:473-474`, `models/replay.py:10-182`).
- `profit_max.py: V2GProfitMaxOracleGB` (l.8-420) is the "Optimal (Offline)" profit-max model, an MIQCP.
- Variables: per-port charge and discharge current, binaries ω_ch and ω_dis, and bilinear `act_current = current·ω` (l.255-267). It has CS and transformer current limits (l.232-253) and energy dynamics with η·V·dt (l.329-333).
- `cs_ch_efficiency = replay.cs_ch_efficiency * 0.9` (l.47-48). The replay sets these to **1** (`replay.py:80-81`), so the oracle assumes η = 0.9 for every EV, while the env uses the per-EV current-dependent tables.
- **The departure energy is SOFT**, a quadratic penalty in the objective (l.353-373):
  ```python
  if t_dep[p, i, t] == 1:
  #     self.m.addLConstr(energy[p, i, t] >= ev_max_energy_at_departure[p, i, t]-5,
  #                       name=f'ev_departure_energy.{p}.{i}.{t}')
      self.m.addConstr(user_satisfaction[p, i, t] == \
          (ev_des_energy[p, i, t] * ev_max_energy[p, i, t-1] - energy[p, i, t])**2,
          name=f'ev_user_satisfaction.{p}.{i}.{t}')
  ...
  self.m.setObjective(costs - 100 * user_satisfaction.sum(), GRB.MAXIMIZE)
  ...
  self.m.params.NonConvex = 2
  ```
  - Units bug: `ev_des_energy` is already kWh (`replay.py:174` stores `ev.desired_capacity`). Multiplying by `ev_max_energy` (kWh) makes the target C², which is unreachable, so the penalty always pushes towards a full battery.
  - The model also writes `model.lp` into the cwd (l.376).
- `v2g_grid.py: V2GProfitMax_Grid_OracleGB` (l.7): the same soft penalty `(ev_des_energy − energy)²` (l.368-375) with a linearised power-flow voltage model. Its objective is 1000·voltage_slack only (l.496).
- `tracking_error.py: PowerTrackingErrorrMin` (l.14): minimises the setpoint tracking error. It has no departure-energy constraint (commented out, l.313-337).
- **Licence**: all of them need gurobipy. Every realistic size (N ≥ 5, T = 96-112) exceeds the 2000-variable restricted licence, so a full or academic Gurobi licence is required.
- **Porting to open source**:
  - The MPC/oracle formulations are plain MILPs (linear objective, linear constraints, one binary per port-step). `NonConvex=2` is set but not needed. They port directly to HiGHS (highspy, Pyomo or cvxpy+HiGHS).
  - The binaries can even be relaxed to an LP when charge and discharge prices are equal and η_c·η_d ≤ 1, because simultaneous charging and discharging is then never strictly profitable. A tie can still yield both, so post-process or keep the binaries.
  - `profit_max.py` becomes a MILP by replacing `act = current·ω` with `0 ≤ act ≤ I_max·ω`, and by replacing the quadratic penalty with the hard constraint `energy[t_dep] ≥ E_des`, or an L1 penalty.
  - **Recommendation**: write our own perfect-information LP/MILP oracle (Pyomo or highspy) in **reported-power units** with hard departure and transformer constraints. Validate it by replaying its actions in EV2Gym.
  - highspy is not installed in the venv yet.

## 9. Metrics (`utils/utils.py:12-123`, `get_statistics`, returned in `info` at episode end)

- Counts and energy:
  - `total_ev_served`;
  - `total_profits` (€, §6);
  - `total_energy_charged` and `total_energy_discharged` (kWh, battery-side = reported).
- Satisfaction:
  - `average_user_satisfaction`: mean of E_dep/E_des capped at 1. It counts as satisfied if E ≥ E_des − 0.001 kWh (`ev.py:211`).
  - `energy_user_satisfaction` (mean, std, min): 100·E_final/`max_energy_AFAP`, where AFAP is the energy reachable when charging flat out at max η without taper (`ev.py:407-440`).
- Tracking:
  - `tracking_error` = Σ(PS − P)²;
  - `energy_tracking_error` = Σ|PS − P|·dt;
  - `power_tracker_violation` = Σ(P − PS)₊. These are meaningless when `power_setpoint_enabled: False`, because PS = 0.
- Constraint indicators:
  - `total_steps_min_emergency_battery_capacity_violation`;
  - `total_transformer_overload` (Σ kW over steps, §5);
  - `battery_degradation`, `battery_degradation_calendar`, `battery_degradation_cycling` (`ev.py:442-521`). This is a semi-empirical calendar plus cycling capacity-loss model with fixed cell parameters (78 kWh reference pack, 2-year age, 298 K), driven by mean SoC, DoD and |energy| throughput;
  - `total_reward`;
  - grid metrics (`voltage_violation`, counters, `saved_grid_energy`) only if `simulate_grid`;
  - `opt_*` fields when a replay with optimal stats is loaded.
- Per-step arrays on env:
  - `current_power_usage[k]`: station EV power, kW reported;
  - `cs_power`, `cs_current`, `tr_overload`, `tr_inflexible_loads`, `tr_solar_power`;
  - `port_energy_level`, `port_current`, `port_current_signal` (not in lightweight mode, which is forced when cs > 100, `ev2gym_env.py:171-172`).
- Per-step `info` contains `action_mask` (1 if an EV is present) and `cost` (`ev2gym_env.py:452-457,488-491`).
- `evaluator.py` (repo root) compares algorithms on common replays: it generates replays with `save_replay=True` and re-runs each algorithm with `load_from_replay_path=...` (`evaluator.py:93-111,166-215`). It needs stable_baselines3, sb3_contrib and torch, which are not installed.

## 10. Scenario controls for experiment design

| Purpose | Key (file:line in V2GProfitMax.yaml) | Notes |
|---|---|---|
| Step / horizon | `timescale` (6), `simulation_length` (7) | 15 min × 112 by default; 5 min works |
| Date | `year/month/day` (13-15), `random_day` (17), `random_hour` (18), `hour/minute` (22-23), `simulation_days` (26) | random_day samples 2022-01-01 + U{0..547} d |
| Arrival profile | `scenario` (29): `public`, `private`(home), `workplace`; also `GF`, which needs `./GF_data/*.npy`, not shipped (`loaders.py:44-53`) | Rate tables `data/distribution-of-arrival*.csv`; energy `mean-demand-per-arrival.csv`; stay `mean-session-length-per.csv` |
| Arrival intensity | `spawn_multiplier` (30) | With workplace, multiplier 5 and 25 chargers: 14-24 EVs per episode (mean 18.7 over seeds 0-29), about ≤1 EV per port per day. The latest departure was step 72 of 112 |
| Min stay | `ev.min_time_of_stay` (147, 180 min) | min stay in steps = 180/15 = 12; observed t_dep − t_arr ≥ 14 |
| V2G | `v2g_enabled` (41) | sets the action low bound and the charger discharge current |
| N chargers | `number_of_charging_stations` (42) | the action and observation dims scale with ports |
| Ports | `number_of_ports_per_cs` (44) | keep 1 (MPC requirement) |
| Transformers | `number_of_transformers` (43), `charging_network_topology` (48, JSON path or None) | topology JSON gives per-transformer `max_power` and per-charger I/V/phases |
| Transformer capacity | `transformer.max_power` (120, 100 kW) | 25 × 22.17 kW = 554 kW nominal (25 × 11 kW = 275 kW EV-limited), so 100 kW is binding |
| Loads / PV / DR | `inflexible_loads` (78-82), `solar_power` (85-89), `demand_response` (92-103) | all off in V2GProfitMax.yaml; multiplier 1 means load peaks at about 100 % of P_tr |
| Grid | `simulate_grid` (50), `network_info` (51-59), `pf_solver` (61) | off |
| Setpoints | `power_setpoint_enabled` (68), `power_setpoint_flexiblity` (69) | off |
| EV fleet | `heterogeneous_ev_specs` (108), `ev_specs_file` (110), `ev.*` (132-151) | homogeneous: config values (11 kW, 50 kWh, η=1, transition_soc=1) |
| Desired energy | `ev.desired_capacity` (149) | fraction of capacity, default 1 |
| Prices | `discharge_price_factor` (35) | |
| Seeds | constructor `seed`, `reset(seed=)`; `tr_seed` (75) | `reset(seed=None)` draws a seed from the global RNG, so the sequence is reproducible from the initial seed |

- Charge-as-fast-as-possible with a non-binding transformer (1000 kW), 20 seeds, V2GProfitMax.yaml: 484 EVs, **99.6 % reached the full-battery target**, min satisfaction 0.989. So individual departure targets are almost always feasible. Joint feasibility under a binding transformer is not guaranteed.
- Env construction takes about 3.5 s (price CSV parse, cached in `env.price_data` for later resets).
- **Common-random-number evaluation**: save replays (`save_replay=True`, `replay_save_path`) once, then build every controller's env with `load_from_replay_path` (same EVs, prices, loads).

## 11. Implications for our method (safety layer and experiments)

1. **Use reported power as the decision variable.**
   - Let p_i [kW] be EV i's power as EV2Gym measures it, i.e. `ΔE_i/dt` = the charger's `current_power_output`. This is exactly what is billed and what is added to the transformer.
   - EV2Gym dynamics are then E_i(k+1) = E_i(k) + p_i·dt exactly (lossless accounting, §3), plus clipping and a +0.01 kWh ceil.
   - The transformer constraint is Σ_i p_i + L_k + PV_k ≤ P_tr,k (and ≥ −P_tr), with L and PV from `tr.inflexible_load[k]` and `tr.solar_power[k]`.
2. **Map p_i to the action** (single port, φ=3, V=400 V, I_max=32 A, so P_cs = 22.17 kW):
   - Charging p_i ≥ 0: a_i = p_i / (η_c,i(I)·P_cs), valid while p_i ≤ P̄_i := η_c,i·min(P_cs, P_ev,i). Any larger action saturates at P̄_i.
   - Discharging p_i < 0: a_i = p_i / (η_d,i(I)·P_cs,dis), valid while |p_i| ≤ P̲_i := η_d,i·min(P_cs,dis, |P_ev,dis,i|).
   - η(I) uses I = |a|·32 A rounded to an integer. With the dict specs, η is piecewise constant in I (for example 0.87 below 9.5 A and 0.90 above for Tesla/VW). This creates **small gaps in the set of achievable p** (for example 5.73-5.92 kW is unreachable) and needs a fixed-point or table inversion.
   - **Recommended**: use scalar efficiencies, either through homogeneous specs or a custom `ev_specs_file` without `3ph_ch_efficiency` keys (then η_c, η_d ~ U[0.95, 1] per EV). The mapping is then linear.
   - **Alternative**: invert by trying each η level and picking the I consistent with round(I). Round the target **up** for the departure bound and re-check the transformer slack.
3. **Linearity within a step**: below the EV limits and the taper threshold, p is exactly linear in a (verified, `t3.py`). The nonlinearities are:
   - **(a) Efficiency steps** (above).
   - **(b) EV power caps**, a saturation that the box bounds handle.
   - **(c) Full-battery cap**: p ≤ (C − E)/dt (SoC clipped to 1).
   - **(d) Discharge floor**: p ≥ −(E − E_min)/dt, which ends a step exactly at `min_battery_capacity`.
   - **(e) CC-CV taper above transition_soc** (on for heterogeneous specs: transition_soc ~ U[0.7, 0.9] and multiplier 5).
   - **(f) +≤0.01 kWh ceil rounding**, favourable for the lower bound.
   - **(g) Out-of-range actions**, which are not rejected.
   - Any request that violates (c), (d) or (e) is silently reduced in magnitude. A clipped *discharge* raises the aggregate power, which can then **exceed the transformer limit** even if the requested profile was feasible. So the safety layer must use the true per-step bounds u_i = min(P̄_i, (C_i − E_i)/dt, P̄_i^taper(E_i)) and l_i ≥ max(−P̲_i, −(E_i − E_min,i)/dt) before the aggregate projection.
4. **Departure bound in EV2Gym units**:
   - At decision step k, τ_i = t_dep,i − k + 1 actions remain, including the current one.
   - With p in reported units, η_c in the user's formula is **1** and P̄ is the reported maximum. The bound becomes l_i = max(−P̲_i, −(E_i − E_min)/dt, (E_req,i − E_i)/dt − (τ_i − 1)·P̄_i), with E_req,i = `ev.desired_capacity` in kWh (absolute; remaining requirement = desired − current).
   - If we want to keep the paper formula with explicit η_c (grid-side p), then EV2Gym's transformer and billing must be changed to grid-side power. That means a runtime subclass or monkeypatch of `EV._charge/_discharge` and `EV_Charger.step` in **our** code, not in the repo, e.g. report P_cmd and debit η_c·P_cmd (charge) or P_cmd/η_d (discharge).
   - **Recommendation**: implement this physical-loss variant as a thin wrapper, so that V2G arbitrage pays a round-trip loss. Otherwise any price spread > 0 is profitable, and the paper must say that EV2Gym's η is only a power derating.
5. **Taper correction**: P̄ is SoC-dependent, so (τ − 1)·P̄ overestimates the reachable energy.
   - Replace it with the max-reachable-energy map R_n(E): the result of n full-pilot steps of `EV._charge` from E. It is monotone in E, so compute it by simulating a deep-copied EV.
   - Then l_i = (R_{τ−1}^{−1}(E_req) − E_i)/dt, computed by bisection or by caching per EV model.
   - Simpler options:
     - (i) homogeneous specs with `transition_soc: 1` (no taper);
     - (ii) a custom spawner or post-spawn override setting `ev.transition_soc = 1`;
     - (iii) E_req ≤ transition_soc·C.
   - With multiplier 5 and a full-battery target, the last 2-3 % SoC takes about 3-4 extra steps. This matters near departure.
6. **Getting the data from env objects at decision step k = `env.current_step`**:
   - For charger j in `env.charging_stations` and port 0, `ev = cs.evs_connected[0]` (None if empty).
   - Fields: `ev.time_of_departure` (t_dep, steps), `ev.time_of_arrival`, `ev.current_capacity` (E, kWh), `ev.desired_capacity` (E_req, kWh), `ev.battery_capacity`, `ev.min_battery_capacity`, `ev.max_ac_charge_power`, `ev.max_discharge_power` (negative), `ev.charge_efficiency`/`discharge_efficiency`, `ev.transition_soc`, `ev.transition_soc_multiplier`.
   - Charger: `cs.get_max_power()`, `cs.get_min_power()`, `cs.connected_transformer`.
   - Transformer: `tr.max_power[k]`, `tr.inflexible_load[k]`, `tr.solar_power[k]`, `tr.get_power_limits(k, H)` (with DR notice).
   - Prices: `env.charge_prices[0,k]` (−λ) and `env.discharge_prices[0,k]`.
   - **Do not read `env.EVs_profiles`** in the agent or safety layer. It holds all future arrivals (oracle information). The MPC baselines use it.
7. **No overstay**: EVs leave exactly after the action at k = t_dep, and the port stays empty for at least 2 steps. With `empty_ports_at_end_of_simulation=True`, int(stay + t + 3) ≤ T − 2 (`utils.py:254-256`), so every EV departs no later than step T − 2 and episode termination never cuts a departure requirement.
8. **dt**: change `timescale` (5, 15, 30 or 60 to keep the load and PV resampling valid) and scale `simulation_length` (288 at 5 min). Arrival rates and stays rescale automatically. Prices stay hourly-constant.
9. **Transformer is never enforced by the env**: our safety layer is the only enforcement. Measure violations with `env.tr_overload` (kW) or our own check, Σp + L + PV − P_tr,k. Do not trust `get_how_overloaded` for reverse-flow violations.
10. **Default scenario sizing**: 100 kW for 25 chargers is binding (275 kW EV-limited demand potential). For the scale-agnostic study, vary N ∈ {10, 25, 50, 100, ...} with P_tr scaled proportionally (for example 4 kW per charger) and keep N_ports = 1. Fix replays per seed for all methods.
11. **Oracle**: build our own Pyomo or highspy LP/MILP with hard departure and transformer constraints in reported-power units (η = 1 accounting, or the physical-loss variant consistent with the wrapper). EV2Gym's Gurobi oracles do not fit: soft departure, units bug, η = 0.9 mismatch, numpy-2 crash, licence limits.

## Questions/uncertainties

1. **Efficiency convention**: should the paper keep EV2Gym's lossless accounting (η as power derating only, with V2G arbitrage that has no round-trip loss) or add a physical-loss wrapper? This decision changes the safety bound (η_c explicit or 1) and the profit numbers.
2. **E_req definition**: EV2Gym's default is a full battery (`desired_capacity = 1.0·C`). Do we use that, a configurable fraction (`ev.desired_capacity` < 1, which changes the arrival-SoC sampling at `utils.py:225-226`), or our own per-EV E_req, e.g. from the sampled required energy? Using < 1 avoids most of the taper region.
3. **Taper**: disable it (homogeneous specs or override `transition_soc=1`) or model it exactly with the R_n map? Heterogeneous specs randomise transition_soc in [0.7, 0.9] regardless of config.
4. **Current-dependent η tables** in `ev_specs_v2g_enabled2024.json` make achievable power non-convex (small gaps). Do we accept the tables (with an inversion routine) or switch to scalar η?
5. **Price timezone**: prices are matched on the naive sim date against UTC hours, so local-time price shapes are shifted 1-2 h. This is probably harmless but should be stated.
6. **Inflexible-load forecast semantics**: `forecast_mean: 30` gives a forecast mean equal to 30 % of the true load (`transformer.py:242-251`). This looks like a bug or odd design. If we use loads, we should use our own forecast (or the true values) in the safety layer, because the safety layer needs L_k for the current step, which is available exactly as `tr.inflexible_load[k]`.
7. **Reverse-flow overload metric** in `get_how_overloaded` measures |P − max| even when P < −max. Should we report our own metric?
8. Whether the pinned commit's Gurobi oracles were ever meant to run under numpy ≥ 2. Even with a full licence, `V2GProfitMaxOracle` needs a small patch (`f[i].item()`), which must not be made in the pinned repo.
9. Multi-port chargers share current through Σ|a| ≤ 1 normalisation, which is not a physical per-port model. We assume 1 port per charger throughout.
10. The ±0.01 kWh ceil gives an unreported energy gain (≤0.01 kWh per active step, ≤ about 0.5 kWh per session). Should we count E-based satisfaction on `current_capacity`, as EV2Gym does, or on Σ p·dt? The safety guarantee should be stated on `current_capacity` (it is ≥ Σ p·dt + E_arr).
11. Joint feasibility: EV2Gym does not guarantee that Σ_i l_i ≤ P_tr − L_k at all times (arrivals are generated without regard to transformer capacity). The method must say what happens when the departure and transformer constraints conflict (priority rule or slack), and the experiments should log such events.
