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

## Where things live

| Concept | Code |
|---|---|
| Stand-in brain shape and wiring; choosing real vs stand-in | `src/brain_model/anatomy.py` |
| Real brain data (download, prepare, load, fMRI) | `src/brain_model/hcp/` |
| Shared hemisphere logic, job patterns | `src/brain_model/hemispheres/base.py` |
| Left / right hemisphere | `src/brain_model/hemispheres/left.py`, `right.py` |
| Corpus callosum + plasticity | `src/brain_model/connectivity/corpus_callosum.py` |
| Disorder (spreading, stroke, tumour, damage map) | `src/brain_model/lesion.py` |
| Healing (use-dependent recovery, golden period) | `src/brain_model/recovery.py` |
| Practice modes, repetition counting, brain stimulation | `src/brain_model/therapy.py` |
| Patient file -> config; problem card | `src/brain_model/patient.py` |
| Target / dose / timing searches | `src/brain_model/planning.py` |
| Patient outcome measures | `src/brain_model/simulation/outcome.py` |
| Plan figures | `src/brain_model/visualization/plan_figures.py` |
| Whole brain | `src/brain_model/brain.py` |
| Experiment and scoring | `src/brain_model/simulation/runner.py` |
| Resting brain + fitting to fMRI | `src/brain_model/simulation/rest.py`, `fit.py` |
| 3D viewer (black screen, two rotatable hemispheres) | `src/brain_model/visualization/` |
| Settings | `configs/default.yaml` (stand-in), `configs/hcp.yaml` (real brain), `configs/care.yaml` (patient care) |
