# Architecture

```text
            stimulus (input vector)
              │                │
              ▼                ▼
     ┌────────────────┐  ┌────────────────┐
     │ Left hemisphere │  │Right hemisphere│
     │ language/logic  │  │ spatial/pattern│
     └───────┬────────┘  └───────┬────────┘
             │◄── corpus ───────►│
             │    callosum       │
             ▼                   ▼
          left state         right state
```

Each time step:

1. Both hemispheres receive the same stimulus and compute their own activity (`process`).
2. The corpus callosum blends each hemisphere's activity with the other's, weighted by `strength`.
   - `strength: 0` gives a "split brain" where the hemispheres never share information.
   - `strength: 1` fully merges them, so both hemispheres end up with identical activity.
   - Values in between keep each hemisphere distinct but influenced by the other.

## Where things live

| Concept | Code |
|---|---|
| Shared hemisphere logic | `src/brain_model/hemispheres/base.py` |
| Left hemisphere | `src/brain_model/hemispheres/left.py` |
| Right hemisphere | `src/brain_model/hemispheres/right.py` |
| Corpus callosum | `src/brain_model/connectivity/corpus_callosum.py` |
| Whole brain | `src/brain_model/brain.py` |
| Simulation loop | `src/brain_model/simulation/runner.py` |
| Settings | `configs/default.yaml` |

The current hemisphere models are simple placeholders (one neural layer each).
Replace them with real AI models as the project develops.
