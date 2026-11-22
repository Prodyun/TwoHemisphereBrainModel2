# Architecture

```text
   left task (sequences)                          right task (patterns)
   + a weaker copy of the right's task            + a weaker copy of the left's task
            │                                              │
            ▼                                              ▼
  ┌──────────────────────┐   corpus callosum   ┌──────────────────────┐
  │   Left hemisphere    │◄───────────────────►│   Right hemisphere   │
  │ 180 regions, mostly  │  region i ↔ mirror  │ 180 regions, more    │
  │ local wiring         │  region i           │ long-range wiring    │
  └──────────┬───────────┘                     └──────────────────────┘
             │
        lesion starts at one region and spreads along the wiring
```

## How one time step works

1. **Callosum signals.** Each region receives a signal from its mirror region on the other side.
   A damaged sender transmits less.
2. **Hemisphere update.** A region's activity moves toward
   `tanh(health × (neighbour input + task input) + callosum input + noise)`.
   Damage reduces a region's own processing. Help arriving through the callosum is not reduced.
   This is the model's key assumption.
3. **Lesion spread.** Damage grows where damage already is and moves to connected regions
   (a network-spread model, like those used for neurodegenerative disease).
   `cross_spread > 0` lets damage leak across the callosum too.
4. **Plasticity (adaptation).**
   - Links into damaged regions strengthen.
   - The healthy hemisphere takes on more of the damaged side's task ("takeover").
   - Takeover only happens when the callosum is connected.

## The experiment

`run_experiment` runs four simulations that all share the same random noise:

| | Healthy | With lesion |
|---|---|---|
| Connected brain | baseline | test |
| Split brain (callosum strength 0) | baseline | control |

**Function score:** how strongly a hemisphere's regions still follow their task input,
compared with the healthy run (100% = as good as healthy). The main result is
`benefit_of_callosum`, the connected score minus the split score for the damaged hemisphere.

## Where things live

| Concept | Code |
|---|---|
| Brain shape, regions, wiring | `src/brain_model/anatomy.py` |
| Shared hemisphere logic | `src/brain_model/hemispheres/base.py` |
| Left hemisphere (sequential task) | `src/brain_model/hemispheres/left.py` |
| Right hemisphere (pattern task) | `src/brain_model/hemispheres/right.py` |
| Corpus callosum + plasticity | `src/brain_model/connectivity/corpus_callosum.py` |
| Disorder that spreads | `src/brain_model/lesion.py` |
| Whole brain | `src/brain_model/brain.py` |
| Simulation, experiment, scoring | `src/brain_model/simulation/runner.py` |
| 3D viewer (black screen, two rotatable hemispheres) | `src/brain_model/visualization/` |
| Settings | `configs/default.yaml` |

## Stand-ins to replace with HCP data

| Now (stand-in) | Later (from HCP-YA) |
|---|---|
| Smooth hand-made hemisphere shape | Real cortical surface (`*.L/R.midthickness.32k_fs_LR.surf.gii`) |
| 180 evenly spread regions | The 180 Glasser regions per hemisphere |
| Distance-based wiring | Diffusion-MRI connectome (within and between hemispheres) |
| Hand-set settings | Tuned so simulated activity matches resting-state fMRI connectivity |
