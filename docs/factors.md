# Adjustable factors

Every factor of the patient-care model (see docs/calibration.md for the formulas used to calibrate them), what it stands for, its default and range, and the source.
The defaults live in `configs/care.yaml`. A patient file can override any of them. Some values
are **assumptions**, because no measured value exists. Those are the ones to sweep and to state on
the poster.

## Time and outcome

| Factor | Meaning | Default | Range / note | Source |
|---|---|---|---|---|
| `time.steps_per_day` | model steps per day | 24 (1 step = 1 h) | assumption | — |
| `care.days` | days simulated after the injury | 120 | covers the CPASS window | Dromerick 2021 PNAS |
| `care.goal` | "recovered" = job back to this share of healthy, over 2 weeks in a row | 0.8 | 0.6–0.9 | choice |
| `care.onset_day` | normal days before the injury | 2 | — | — |

## Healing (`recovery`)

| Factor | Meaning | Default | Range / note | Source |
|---|---|---|---|---|
| `spontaneous_per_day` | share of healable damage repaired per day without practice | 0.16 | **calibrated** | van der Vliet 2020 Ann Neurol recovery curves (docs/calibration.md) |
| `use_rate` | extra healing per practice step per unit of use | 0.3 | **weakly constrained**: top of the range tried; 0.03 fits almost as well | van der Vliet 2020; CPASS 2021 |
| `callosal_use_weight` | matching input from the healthy side also counts as use | 1.0 | **supported** (correlational), 0 = only the area's own activity | Yuan 2020; Paul 2023; Kokinovic & Medini 2018 |
| `floor_fraction` | share of the worst damage that is permanent (unless practice lowers it) | 0.3 | **calibrated** | van der Vliet 2020 (recovery coefficients r_k) |
| `floor_use_rate` | how fast practice lowers the permanent floor (raises the final level, not only the speed) | 0.03 | **calibrated** to CPASS, but CPASS still not matched (1.6% vs 15.8%); new rule, 0 = old behaviour | Dromerick 2021 PNAS (CPASS) |
| `saturation_reps` | reps per session after which more reps help half as much | 300 | 100–1000 | no dose-response from 3,200 to 9,600 reps: Lang 2016 Ann Neurol |
| `window_points` | golden period: (days, plasticity gain) | (0, 0.5) (20, 0.76) (75, 1.0) (180, 0.35) (365, 0.2) | the decay rate has never been measured in humans: sweep it | CPASS: +5.25 / +6.87 / +2.41 ARAT (Dromerick 2021) |

## Practice (`therapy`)

| Factor | Meaning | Default | Range / note | Source |
|---|---|---|---|---|
| `mode` | physical, imagery, action_observation, bci (mixes allowed), plus sleep_replay, vns | physical | — | — |
| `start_day`, `days_per_week`, `session_steps` | schedule | 7, 5, 2 h | — | typical rehab |
| `reps_per_step` | repetitions per practice hour | 150 (300 per session) | 100–300 per session | VNS-REHAB >300 per session (Dawson 2021 Lancet); Lang 2016 |
| `imagery.primary` / `imagery.other` | imagery input in primary motor/touch areas / other job areas | 0.35 / 0.75 | M1 0.2–0.5, premotor/parietal 0.6–0.9 (assumed; no measured "% of execution") | Hardwick 2018 Neurosci Biobehav Rev |
| `action_observation.*` | the same, for watching a movement | 0.3 / 0.6 | — | Hardwick 2018 |
| `feedback` | sensory feedback when the limb really moves (or FES moves it) | 0.3 | — | assumption |
| `bci.threshold` | decoder detects "trying" above this activity | 0.1 | — | — |
| `bci.contingent` / `hebbian_bonus` | feedback exactly when trying doubles healing | true / 1.0 | contingent vs random: +16.2 vs +4.8 FMA | Sci Rep 2024 (n=20 completers) |
| `bci.decoder` | read the damaged or the healthy side (IpsiHand) | damaged | — | IpsiHand FDA De Novo 2021 |
| `sleep_replay.strength`, `steps` | night-time replay of the practised pattern | 0.1, 2 h | sweep 0–0.2: pilot evidence only | Cheng 2021 J Neurosci; NCT04312269 |
| `vns.multiplier` | healing multiplier for paired VNS | 2.1 | 1.8–2.9 (rats); 0 effect if unpaired | Dawson 2021 Lancet; Khodaparast 2016; Hays 2014 |

## Stimulation (`stimulation`)

| Factor | Meaning | Default | Range / note | Source |
|---|---|---|---|---|
| `targets` | areas and strengths; +0.5 = 50% more excitable, -0.3 = 30% calmer | chosen by the target finder | calming capped at -0.6 | — |
| `paired` | only during practice | true | — | — |
| `focality_mm` | how far the coil's effect spreads to neighbouring areas | 12 mm | 0–25 mm (figure-8 coil ~2–3 cm) | assumption |
| `callosal_output` | stimulated areas also send more across the callosum | true | — | transcallosal facilitation from contralesional PMd: Bestmann 2010 J Neurosci |
| standard targets | today's comparison: calm the healthy M1 (motor), calm right IFG 45 (language) | care.yaml | — | NICHE (Harvey 2018); E-FIT (2023); Ren 2014 |

## Corpus callosum

| Factor | Meaning | Default | Source |
|---|---|---|---|
| `strength` | link strength (0 = split brain, the control) | 0.381 (fitted to resting fMRI) | this project's fit |
| `plasticity_rate` | how fast the bridge adapts after damage | 0.00888 in patient runs (0.02 in hcp.yaml fMRI runs) | **calibrated**: 95% by day 14, Kokinovic & Medini 2018 |
| `plasticity_gain`, `takeover_gain` | how much the bridge strengthens / the healthy side takes over | 1.5, 3.0 | assumption (DCM sign pattern used as a check) |
| `inhibition` | the interhemispheric "brake" (movement areas only via `inhibition_jobs`) | 0 (off) | **checked**: no value passed NICHE + Di Pino + DCM; k = (1/IHI − 1)/a once Murase 2004's ratio is filled in |

## Diseases (`configs/diseases/*.yaml`, one tab each; see docs/diseases.md)

| Engine (`engine.type`) | Behaviour | Factors |
|---|---|---|
| stroke | the core is hit at once; the surrounding at-risk tissue (penumbra) starts at `penumbra_start` (30%) of its final damage and dies off over `penumbra_days`; no spread along the wiring | `penumbra_days` 5, `penumbra_start` 0.3 (assumption; real penumbra is lost over hours to days) |
| tumor | the centre grows to max over `growth_steps`, with pressure on neighbours | `growth_steps` 480 (20 days), `pressure` 0.3 |
| spreading | travels along the wiring (network Fisher-Kolmogorov); seeds can be on both sides | care runs: `transport` 0.001, `growth` 0.0016 (~12 areas by day 60, ~60 by day 120); `cross_spread` (0 = the bridge does not carry it); the 960-step experiment keeps 0.002 / 0.0105 |
| epilepsy | an over-excitable focus with random seizures | `excitability` 1.6 (sweep 1.2–2.0), `seizures_per_day` 1, `seizure_drive` 2.5, `seizure_steps` 1 (one step = one seizure): all assumptions |

`callosal_injury` (MS, TBI; `connectivity/callosal_injury.py`): every number below is an assumption with a sweep range. The papers show that the callosum is damaged (Evangelou 2000, Ozturk 2010, Johnson 2013), not how fast.

| Factor | MS default | TBI default | Meaning |
|---|---|---|---|
| `kind` | relapsing | hit | random relapses vs. one injury |
| `relapses_per_year` | 1.0 (0.3–1.5) | – | relapse rate |
| `areas_per_relapse` / `fraction` | 60 areas | 60% of areas | how much of the bridge one hit reaches |
| `severity` | 0.6 | 0.4 | share of an area's callosal input lost |
| `recover_fraction`, `recover_days` | 0.5, 45 | 0.4, 90 | repair (remyelination) |
| `progressive_per_year` | 0.1 | – | slow loss between relapses |

## Healthy side helps (`care.yaml: bridge`; `connectivity/bridge_therapy.py`, `healthy_side.py`)

| Factor | Meaning | Default | Range / source |
|---|---|---|---|
| `pair.rate` | Bridge Builder: share of the remaining gain added per paired session step | 0.05 | assumption, sweep 0.01–0.1 |
| `pair.max_gain` | cap on the extra fibre strength (× the average measured fibre) | 2.0 | assumption, 0.5–3 |
| `pair.decay_per_day` | fading when not refreshed | 0.01 | assumption; ccPAS after-effects are short per session (Falaschi 2025 review) |
| `pair.order` | forward (healthy then damaged) strengthens; reverse weakens (the control) | forward | Hebbian timing rule of ccPAS |
| `mirror_drive.gain` | stimulation per unit of healthy-side activity | 2.0 | assumption, 0.5–3 |
| `mirror_drive.threshold` | minimum healthy-side activity that triggers | 0.05 | assumption |
| `mirror_drive.timing` | closed loop, or the same dose at fixed (open) or random times | closed | controls get the closed loop's measured share |
| `stimulation.taper` | Hand-off: `{kind: step / linear / exp, from_day, to_day}` | none | therapy.taper_factor |
| `care.meaningful_change` | a plan only "wins" if it reaches the goal sooner or ends ≥ 1 point higher | 0.01 | assumption |

## Intake and spread risk

| Factor | Meaning | Default | Source |
|---|---|---|---|
| mask overlap | areas with < 2% of their voxels in the lesion are ignored | 2% | assumption |
| tau-PET `cutoff` | SUVR above which an area counts as having tau | 1.2 | assumption; published cut-offs ~1.2–1.4 depend on tracer and region |
| EEG symmetry index | `|L − R| / (L + R)` of slow-wave power; ≥ 0.1 = one side | 0.1 | ELECTRA-STROKE (delta BSI AUC 0.91) |
| risk ensemble | runs, and spread speed range | 30 runs, 0.5–1.5× | assumption |
| "reached" | damage at or above | 10% | as in the viewer |

Still to come in later phases: containment, the detection markers, patient biomarkers, the
area-atlas genes and receptors, and the brain-reserve scaling.
