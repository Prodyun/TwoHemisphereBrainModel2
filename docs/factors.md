# Adjustable factors

Every factor of the patient-care model, what it stands for, its default and range, and the source.
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
| `spontaneous_per_day` | share of healable damage repaired per day without practice | 0.003 | 0–0.01; to be calibrated | proportional-recovery debate (Chong 2023 NNR; Bowman 2021 Stroke) |
| `use_rate` | extra healing per practice step per unit of use | 0.03 | to be calibrated (phase 3) | — |
| `callosal_use_weight` | matching input from the healthy side also counts as use | 1.0 | 0 = only the area's own activity | Hebbian assumption; callosal plasticity |
| `floor_fraction` | share of the worst damage that is permanent | 0.3 | 0–0.6 | assumption |
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
| `plasticity_rate`, `plasticity_gain`, `takeover_gain` | adaptation after damage | 0.02, 1.5, 3.0 | configs/hcp.yaml |
| `inhibition` | the interhemispheric "brake" | 0 (off) | bimodal balance model: Di Pino 2014 Nat Rev Neurol |

## Disorders (`care.yaml: disorders`)

| Disorder | Behaviour | Factors |
|---|---|---|
| stroke | the damage map appears at once and does not spread | — |
| tumor | the centre grows to max over `growth_steps`, with pressure on neighbours | `growth_steps` 480 (20 days), `pressure` 0.3 |
| spreading | travels along the wiring (network Fisher-Kolmogorov) | `transport` 0.002, `growth` 0.0105 |

Still to come in later phases: containment, the detection markers, patient biomarkers, the
area-atlas genes and receptors, and the brain-reserve scaling.
