# Architecture

```text
 left jobs: language, math/sequences,          right jobs: spatial attention, faces,
 vision + movement (right side of body)        emotion in voice, vision + movement (left side)
            │                                              │
            ▼                                              ▼
  ┌──────────────────────┐   corpus callosum   ┌──────────────────────┐
  │   Left hemisphere    │◄───────────────────►│   Right hemisphere   │
  │ 180 Glasser areas,   │  real measured      │ 180 Glasser areas,   │
  │ its own real wiring  │  left-right fibres  │ its own real wiring  │
  └──────────┬───────────┘                     └──────────────────────┘
             │
        a disorder starts at one area (e.g. Broca's area) and spreads along the wiring
```

## Two distinct hemispheres that share some qualities

| | Left | Right | Shared |
|---|---|---|---|
| Wiring inside the hemisphere | real left-left connectome | real right-right connectome | same atlas (180 Glasser areas per side) |
| Settings (leak, gain, noise) | can be set/fitted separately | can be set/fitted separately | same equation |
| Jobs (functions) | language, math/sequences | spatial attention, faces, emotion/prosody | vision and movement, each for the opposite half of the body/visual field |
| Callosum | receives real right→left fibres | receives real left→right fibres | |

The real brain is used when `anatomy.source: hcp` (`configs/hcp.yaml`). The original hand-made
stand-in (`anatomy.source: synthetic`, `configs/default.yaml`) still works and gives exactly the
same results as before.

## How one time step works

1. **Callosum signals.** Each area receives signals from the areas it is wired to on the other side
   (`corpus_callosum.wiring`: `real` measured fibres, `mirror` = only the matching area, or `both`).
   A damaged sender transmits less.
2. **Hemisphere update.** An area's activity moves toward
   `tanh(health × (neighbour input + job input) + callosum input + noise)`.
   Damage reduces an area's own processing. Help arriving through the callosum is not reduced.
   This is the model's key assumption.
3. **Jobs.** Each job drives its own areas with its own time pattern: a front-to-back pulse
   sequence (language), a counting ramp (math), smooth waves (spatial attention), bursts (faces),
   slow rhythms (prosody, vision, movement). Each hemisphere also does a weaker copy of the other
   side's jobs at the matching areas (`shared_task`).
4. **Disorder.** `lesion.type: spreading` moves along the wiring. The real brain uses
   `model: fisher_kpp`, the network Fisher-Kolmogorov model of prion-like protein spread
   (Weickenmeier et al. 2019; Schafer et al. 2021): each step a little damage is carried along each
   fibre (`transport`, shorter fibres carry more) and then grows where it lands (`growth`). Because
   transport is slow and growth fast, areas are reached one after another, by wiring distance from
   the start: a front that travels outward (with L_44 as the start, direct neighbours are reached
   around step 440, the next ring ~730, the next ~870). `model: simple` is the earlier rule, in which
   every connected area rises together. `stroke` damages the start area and its most strongly linked
   neighbours once, then stops. `cross_spread > 0` lets a spreading disorder leak across the callosum.
5. **Plasticity (adaptation).** Links into damaged areas strengthen, and the healthy hemisphere
   takes on more of the damaged side's jobs ("takeover"), only when the callosum is connected.

## The experiment

`run_experiment` runs four simulations that all share the same random noise:

| | Healthy | With disorder |
|---|---|---|
| Connected brain | baseline | test |
| Split brain (callosum strength 0) | baseline | control |

**Function score:** how strongly a hemisphere's areas still follow their job input, compared with
the healthy run (100% = as good as healthy). It is reported for the whole hemisphere and for each
job separately. The main result is `benefit_of_callosum` = connected minus split score for the
damaged hemisphere. `scripts/run_batch.py` repeats this over 20 seeds and 20 start areas.

## Checking the model against real brains

`scripts/fit_to_fmri.py` runs the model at rest (no jobs, no damage) and compares its functional
connectivity (FC, how much each pair of areas rises and falls together) with real resting-state
fMRI from the HCP group and the pilot subjects. Left and right settings are fitted separately.
Controls: shuffled wiring (same number of links per area, random partners) and identical left/right
settings. `simulation/rest.py` is a fast version of the same equation for many settings at once,
and a test checks it matches the full model exactly.

What the fit found (HCP group, `results/fit_20261122_000451`):
- Real wiring matches real resting activity far better than shuffled wiring
  (0.256 vs 0.023; with new noise 0.256 vs -0.005).
- Separate left/right settings did **not** fit better than identical ones (held-out 0.256 vs 0.257).
  At group level the hemispheres differ through their wiring and jobs, not their settings.
- The model does about as well as the wiring alone (0.261), which is typical for simple models.
  Search runs gave different "between hemispheres" scores, so the fit landscape is flat and the
  fitted numbers are only loosely determined.
- The fitted settings are in `configs/hcp.yaml`. With them, the brain sits closer to its tipping point
  (gain about 0.85-0.89), so areas pass more signal along.

Note on scores: a function score can go slightly **above 100%** when the healthy side sends its own copy
of the job back across (takeover). The function is then carried partly by the other hemisphere.

## Support from the healthy side, for every start area (scripts/support_map.py)
Every one of the 360 areas was used as the start (3 seeds each; help = connected minus split brain).
The code is left/right symmetric (tested on a mirror-image brain: identical help), so differences
below come from the real wiring, the fitted settings and each side's jobs.
- Left hemisphere damaged: the right side helps from all 180 start areas, mean +8.2 points
  (95% CI +7.7 to +8.8). Most help when the start is in motor/touch areas (4, 1, 3b, 6d);
  least in the temporal language areas along the STS.
- Right hemisphere damaged: the left side helps less, mean +3.3 (+3.0 to +3.7); 159 areas
  helped, 21 no real effect, none hurt overall.
- Interference finding: when the right is damaged, its **vision** gets worse with the callosum
  connected (-3.8 points, 142 of 180 starts), because the left hemisphere sends its own, different
  half of the visual field. Each side's language/prosody can also be slightly hurt from a few starts.
  This is a model result, not a tuned one.

## Patient care: from a patient's damage to a personal plan

The model can now **heal** as well as be damaged. The project's question, *can the healthy
hemisphere help the damaged one recover, and how can we make that faster and personal?*, becomes
a pipeline (`scripts/therapy_plan.py`):

1. **Identify** (`patient.py`): the patient's damage map is placed on the 180 areas per side
   (`lesion.damage_map`, either or both hemispheres). The problem card lists the epicentre, the
   damaged areas and the jobs that lost the most.
2. **Recruit** (`planning.search_targets`): the healthy-side areas best placed to reach the damage
   (real callosal fibres, plus one step through the healthy side's own wiring) are each stimulated
   during practice, made more excitable and calmer. They are ranked against practice alone, today's
   standard target, random targets and the split brain.
3. **Dose** (`therapy.py`): which practice to use, crossed with how strong the stimulation is.
   - physical
   - imagery
   - action observation
   - BCI with contingent feedback
   - mixes, with sleep replay and paired VNS as add-ons
4. **Time** (`planning.search_timing`): the same plan started on different days.
5. **Predict** (`simulation/outcome.py`): weekly function of the trained job, scored only outside
   practice, so practice input never counts as recovery.

How healing works (`recovery.py`):

    damage -= window(days) * (spontaneous + use_rate * boost * saturation * use) * (damage - floor)

- *use* is Hebbian: the area's activity that moves together with the practised movement, plus the
  matching input arriving from the healthy side through the corpus callosum. Help from the healthy
  side therefore drives **repair**, not just cover.
- *window* is the golden period, from the CPASS trial.
- *saturation* means more repetitions in one session give diminishing returns (Lang 2016).
- *boost* comes from paired VNS or contingent BCI feedback.
- *floor* is permanent damage.

Stimulation multiplies an area's excitability, spreads to neighbours as a TMS coil does
(focality ~12 mm), and raises what the area sends across the callosum (transcallosal facilitation).
An optional callosal **brake** (`corpus_callosum.inhibition`) lets each side turn down the other
(interhemispheric inhibition). It is off by default.

First findings on the example patients (they are predictions, and will be calibrated in the
validation phase):
- **Calming the healthy M1** (today's standard rTMS) does not help a severe motor stroke. This matches
  the null NICHE and E-FIT trials.
- **Making the healthy-side areas that reach the damage more excitable** helps a little.
- Combining no-movement practice (BCI) with VNS and stimulation reached the goal with **no physical
  repetitions**, faster than standard care.
- In the split brain the same plan never reaches the goal.
- A growing tumour cannot be out-practised. It must be contained first (next phase).

## The AI Help Predictor (per-area chance of help, and how)

- **Training data** (`help_data.py`, `scripts/build_help_dataset.py`): 7,200 simulated patients.
  - Every one of the 360 areas is damaged, together with its 4 most strongly wired neighbours.
  - Severity 0.6 or 0.9; stroke or tumour; 5 practice modes.
  - Each case runs connected and split with the same noise.
  - Labels: help (connected minus split final function), reached the goal, weeks, physical repetitions.
  - Areas outside any configured job get a *local job* (`patient.ensure_job`), so every area can be scored.
- **Features** (`help_model.area_features`): real properties of the damaged area.
  - callosal fibres and their strength, a fibre to the mirror area, homotopic resting FC, two-step reach (HCP)
  - network (Cole-Anticevic), volume, position
  - plasticity and vulnerability genes (Allen atlas), 7 receptor densities (Hansen 2022 PET)
  - plus the case: severity, disorder, mode
- **Model:** scikit-learn HistGradientBoosting. A classifier predicts whether the goal is reached; regressors predict help and final function.
- **Honest testing:** 5-fold cross-validation **grouped by area**, so every score is on areas the model never saw. It is compared with a wiring-only model and with always guessing the average. Permutation importance gives the "why".
- **Results (2026-11-23):**

  | Measure | AI | Baseline |
  |---|---|---|
  | AUC for reaching the goal | 0.94 | 0.86 (wiring only) |
  | R² for help | 0.75 | 0.08 (wiring only) |
  | Mean error for help | 1.7 points | 4.7 (always the average) |

  Hypothesis H1 in `docs/novelty.md` passes.
  - Most help: early visual areas on the left, orbitofrontal areas, and the motor strip.
  - Least help: the left temporal language areas along the STS, which matches the support map.
  - Negative help: the right early visual areas. This is a consequence of giving each side's vision job a different rhythm, not a finding.
- **The "how"** (`help_pathways.trace`): during practice, the help reaching damaged area i from healthy area j is gain × callosal route weight × j's activity matched to the practised pattern. Two-step routes through a damaged-side neighbour are included. The result is each helper's share and route type (direct fibre / two steps via X).
- **Live demo:** `care_service.py` + `visualization/care_view.py` + the viewer's patient-care mode (AI panel, help arcs, therapy banner, weekly chart).

## The Clinic: every disease, an uploaded patient, and the healthy side as a treatment

The explorer (`scripts/explore.py`) has one tab per disease, built from `configs/diseases/*.yaml` by
`diseases.py` (docs/diseases.md). For any patient it answers, in order:

1. **Where** (`intake.py`): the damage on the 180 areas per side, from one of:
   - a lesion mask in MNI space, overlapped with the volumetric Glasser atlas;
   - an area table (damage, or tau-PET SUVR);
   - a no-scan symptom check plus an EEG symmetry index.
2. **What spreads next** (`spread_risk.py`): an ensemble of untreated spreads with uncertain speeds.
   It gives the chance that each area is reached by day 30, 90 or 180, plus the next areas.
   Stroke shows its at-risk ring, MS/TBI the bridge fibres at risk, epilepsy the areas seizures recruit.
3. **Which treatment is fastest** (`care_service.options`): every option is raced and ranked by weeks to
   the goal. Each one carries a badge from `configs/evidence.yaml` (FDA approved / in trials / model idea).
   The race also gives a dose curve for the helper area, with the smallest dose that works.
4. **The healthy side treats the damaged side** (`connectivity/bridge_therapy.py`, `healthy_side.py`):
   - *Bridge Builder*: paired stimulation (ccPAS) strengthens one chosen callosal fibre. It is compared
     with the reverse order, a random sender and the split brain.
   - *Mirror Drive*: healthy-side activity triggers stimulation of the damaged side in a closed loop.
     It is compared with the same dose given at fixed or random times.
   - *Help-or-Hinder*: EXCITE, CALM or LEAVE the healthy side.
   - *Hand-off*: weaning schedules, scored by function after the help stops.

New disease engines:

- epilepsy: an excitable focus with seizures (`lesion.py`, plus `excitability` / `seizure_drive` in `hemispheres/base.py`);
- damage to the bridge itself (`connectivity/callosal_injury.py`): MS relapses and TBI axonal injury;
- multi-seed spreading on both sides.

All defaults leave earlier runs unchanged.

`scripts/validate_trials.py` (R-EXP-10) checks virtual cohorts against three trials:

- NICHE/E-FIT: an independent test;
- VNS-REHAB and CPASS: calibration checks, because their numbers set the model.

## Assumptions and limitations (say these on the poster)

- The wiring is a **group average** of many people, not one person's brain.
- Diffusion tractography misses many callosal fibres: 78 of 180 left areas (including Broca's area)
  have no detected direct link to the right side. That is why mirror and real+mirror callosums
  are also tested.
- The template brain shape is nearly symmetric. Real individual asymmetry is future work.
- Each area is a population average, not individual spiking neurons.
- The jobs' area lists come from the literature, not from these subjects.
- BOLD stand-in: fMRI measures slow blood flow, so simulated activity is smoothed before
  computing FC. This is a simplification.
- The spreading settings (transport 0.002, growth 0.0105) are chosen so about half the hemisphere
  is damaged by the end of a run. They are not fitted to patient data. Fibre length is approximated
  by the straight-line distance between area centres (real fibres curve, so they are longer).
- The model has 180 areas per hemisphere; the 3D surface (~32,000 points) is only drawn on. Every
  point in an area shows that area's single value, which is why the viewer outlines each area.
- 2 subjects is a pilot, not proof.
- Patient care is a model, not medicine. Several settings are literature-based assumptions,
  listed in docs/factors.md:
  - the time scale (1 step = 1 hour)
  - how strong imagery is
  - the VNS multiplier
  - the BCI bonus
  - sleep replay
  - how fast the golden period closes
- For a language patient, "physical" practice means speaking out loud and "imagery" means
  inner speech.
- Clinic diseases other than stroke and tumour are simulation only (the AI was not trained on them).
  Their time is compressed, the Braak stages are mapped to Glasser areas by this project, the model has no
  thalamus, basal ganglia or spinal cord, and the MS/TBI/epilepsy rates are assumptions (docs/factors.md).
- One step is one hour, so Bridge Builder keeps only the ORDER of the paired pulses (not their millisecond
  gap), and one epilepsy step stands for one seizure.

## Where things live

| Concept | Code |
|---|---|
| Stand-in brain shape and wiring; choosing real vs stand-in | `src/brain_model/anatomy.py` |
| Real brain data (download, prepare, load, fMRI) | `src/brain_model/hcp/` |
| Shared hemisphere logic, job patterns | `src/brain_model/hemispheres/base.py` |
| Left / right hemisphere | `src/brain_model/hemispheres/left.py`, `right.py` |
| Corpus callosum + plasticity | `src/brain_model/connectivity/corpus_callosum.py` |
| Disorder (spreading, stroke, tumour, epilepsy, damage map) | `src/brain_model/lesion.py` |
| Healing (use-dependent recovery, golden period) | `src/brain_model/recovery.py` |
| Practice modes, repetition counting, brain stimulation | `src/brain_model/therapy.py` |
| Patient file -> config; problem card | `src/brain_model/patient.py` |
| Target / dose / timing searches | `src/brain_model/planning.py` |
| Patient outcome measures | `src/brain_model/simulation/outcome.py` |
| Plan figures | `src/brain_model/visualization/plan_figures.py` |
| Help pathways (how the healthy side helps) | `src/brain_model/help_pathways.py` |
| AI Help Predictor: data, features, model | `src/brain_model/help_data.py`, `help_model.py` |
| Live demo logic; viewer data for care | `src/brain_model/care_service.py`, `visualization/care_view.py` |
| Disease registry (one tab per disease) | `src/brain_model/diseases.py`, `configs/diseases/*.yaml` |
| Patient intake (mask / table / symptoms + EEG) | `src/brain_model/intake.py` |
| Spread-risk map | `src/brain_model/spread_risk.py` |
| Bridge Builder, Mirror Drive (mechanisms) | `src/brain_model/connectivity/bridge_therapy.py` |
| Help-or-Hinder, Hand-off, pair search (plans) | `src/brain_model/healthy_side.py` |
| Damage to the bridge (MS, TBI) | `src/brain_model/connectivity/callosal_injury.py` |
| Treatment options mapped to devices and trials | `configs/evidence.yaml` |
| Validation against trials | `scripts/validate_trials.py` |
| Whole brain | `src/brain_model/brain.py` |
| Experiment and scoring | `src/brain_model/simulation/runner.py` |
| Resting brain + fitting to fMRI | `src/brain_model/simulation/rest.py`, `fit.py` |
| 3D viewer (black screen, two rotatable hemispheres) | `src/brain_model/visualization/` |
| Settings | `configs/default.yaml` (stand-in), `configs/hcp.yaml` (real brain), `configs/care.yaml` (patient care) |
