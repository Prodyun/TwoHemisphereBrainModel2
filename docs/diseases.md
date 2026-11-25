# Diseases in the Clinic (one explorer tab each)

Each disease is one file in `configs/diseases/`. The explorer builds a tab from every file it finds, so you
can add a disease in two ways:

1. **In the app:** open *Build a new disease ▸*, click the area where it starts, pick how it behaves, then
   press *Save as a new tab*. This writes `configs/diseases/custom_<name>.yaml`. The tab is marked
   "uncited" until you add sources.
2. **By hand:** copy a file below, change it, and restart `scripts/explore.py`.

## The tabs

| Tab | How it behaves in the model | What it shows | Biggest limit |
|---|---|---|---|
| Stroke | sudden damage plus a dying at-risk ring (penumbra) | which healthy-side areas to stimulate; weeks saved | example damage maps |
| Tumour | slow growth plus pressure on neighbours | slow damage gives the healthy side time | growth speed assumed |
| Alzheimer's | tau spreads from EC/PeEc along the wiring (Braak stages) | the help from the other side vs. the bridge also carrying tau | time compressed |
| Parkinson's | cortical stage only (Braak PD 4–6), one-sided onset | the left-right difference as an early sign | no substantia nigra or basal ganglia |
| ALS | motor cortex on both sides | the honest limit: no healthy side | cortex only |
| MS | relapses damage the callosal fibres; partial repair | how the help changes as the bridge weakens | relapse numbers assumed |
| TBI | a bruise, scattered damage, and a callosal hit | recovery with a hurt bridge | injury sizes assumed |
| Epilepsy | an over-excitable focus and seizures | how much of a seizure crosses the bridge (little, in this model) | no thalamus |

What the model found so far (example patients, simulation only):

- **Alzheimer's:** memory ends at about 72% with the bridge vs. 47% split. If tau also crosses the bridge
  (`cross_spread` 0.5), it ends at 46%, which cancels the help.
- **ALS:** both sides decline and no plan reaches the goal.
- **Epilepsy:** seizures barely cross the callosum here. The crossing is about the same as the noise level
  seen with the bridge cut, because real seizures generalise through the thalamus, which the model lacks.
- **MS / TBI:** partly damaging the bridge changed little, and was even slightly better than an intact bridge,
  while cutting it fully was worst. A likely reason is that a weaker bridge carries less cross-talk from the
  other side's own rhythm but still some help. This is a finding to report with care.

## Writing a profile

```yaml
name: my_disease          # file and tab id (letters, digits, _)
tab: "My disease"         # tab label; tab: null hides it
icon: "✚"
order: 9                  # tab position
ai: false                 # true only for stroke/tumour (the AI was trained on those)
default_area: L_6d        # where it starts if nobody clicks
severity: 0.7             # 0.05 .. 0.98
seed:
  mode: clicked           # clicked: the area + its `neighbors` strongest same-side links
  neighbors: 4
  neighbor_share: 0.6
  # mode: areas  ->  areas: {EC: 0.6, PeEc: 0.5}, sides: {left: 1.0, right: 0.8}
  # diffuse: {level: 0.15, share: 0.3}   (small damage on 30% of all areas)
engine: {type: stroke}    # stroke | tumor | spreading | epilepsy, plus that engine's settings (docs/factors.md)
job: movement             # optional: the job to train and score
extra_jobs: {}            # optional: e.g. a memory job (areas must not be used by another job)
config: {}                # optional overrides, e.g. care: {days: 180}, callosal_injury: {...}
stages: []                # optional: [{name, areas}] -> "when is each stage reached"
plans:                    # the treatments to compare (a split-brain control is always added)
  - {key: standard_care, label: "Practice", changes: {therapy: {mode: physical}}}
  - key: helped
    label: "Practice + excite the healthy partner"
    changes: {stimulation: {targets: [{area: "@mirror", strength: 0.5}]}}   # @mirror / @epicentre
sliders:                  # the what-if controls in the tab
  - {key: severity, label: "How bad", min: 0.3, max: 0.95, step: 0.05}
plain:                    # the simple view: short sentences
  what: "..."
  simulate: "..."
  shows: "..."
  use: "..."
  limits: "..."
clinician: {scale: "...", today: "..."}
assumptions: ["..."]
sources: ["Author et al. year Journal"]
```

Every number in a profile should either cite a source or be marked as an ASSUMPTION with a range to sweep.
