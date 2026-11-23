# What is new in this project (prior-art check and pre-stated hypotheses)

*Written on 2026-11-22, BEFORE the AI model's results were known.*

**Search coverage:**
- Sources: PubMed/PMC, bioRxiv/medRxiv, arXiv, journal sites, and the Society for Science ISEF abstract database (2014–2026).
- Not fully searchable: conference abstract books (OHBM, ISC, SfN, BCI Society) and Regeneron STS (no public archive).

**How to say it:** "to our knowledge, based on a search of PubMed, bioRxiv/medRxiv, arXiv and the ISEF abstract database (2014–2026)". Never "first ever".

## Gaps, the closest existing work, and what this project adds

| # | Gap | Closest existing work | What it did not do | Verdict |
|---|---|---|---|---|
| G1 | For EACH brain area, the chance that the healthy hemisphere can help | Chakraborty, Deco, Roy et al. 2023, *Cereb Cortex Commun* ([PMC10409568](https://pmc.ncbi.nlm.nih.gov/articles/PMC10409568/)): virtual lesions in 49 connectomes; compensating areas predicted by structural and dynamical similarity | a few lesion sites only (68-area atlas); no machine-learning surrogate, no healing, no therapy | **Novel** |
| G2 | Personal healthy-side stimulation targets from a whole-brain simulation, compared with today's target | BrainX3, *Front Neurol* 2023 ([PMC10719856](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10719856/)), a proposal with no results; network-control target search in healthy brains (*Brain Stimul* 2022); E-FIT / NICHE (calming the healthy M1 does not beat sham) | no traced help pathways; no test against the failed standard target | **Novel (application)** |
| G3 | How many physical repetitions practice without moving saves (imagery, contingent BCI, sleep replay, paired VNS) and their combinations | Han, Arbib, Schweighofer 2008, *PLoS Comput Biol*; Norman, Wolpaw, Reinkensmeyer 2022, *Brain Commun* (two-hemisphere network model in which targeting PMd helped and M1 was maladaptive) | no model of VNS, sleep replay or contingent BCI; no "repetitions saved"; no combinations | **Novel** |
| G4 | A plasticity "golden period" for each area | single-area timing simulation (*J Neurophysiol* 2014); criticality recovery (Rocha 2022, *Nat Commun*) | no per-area window across the whole brain | **Novel per area** |
| G5 | A stroke "digital twin" that outputs a treatment plan | BrainX3 (proposal); an aphasia outcome twin (medRxiv 2026); virtual brain twins (*Natl Sci Rev* 2024, no stroke chapter) | no mechanistic twin that outputs where, which therapy and when | **Partly done: we extend it** |
| G6 | A real person's EEG (BCI) driving a simulated recovering brain | closed-loop motor-imagery EEG simulator (*Front Hum Neurosci* 2022), which runs the other way: it simulates the user | the reverse direction (real EEG drives a simulated patient) | **Novel** |
| G7 | The corpus callosum as both a helper and a highway for disease, in one model | network-diffusion spread along the callosum (*PLoS Comput Biol* 2017); conceptual frameworks (*J Pers Med* 2022; Di Pino 2014) | no simulation with both roles and their trade-off | **Novel** |

Past ISEF projects (2014–2026) on stroke include:
- BCI or exoskeleton hardware (2018, 2025, 2026)
- an aphasia BCI (2023)
- a Virtual Brain vision/DBS simulator (2023)

None simulates compensation by the healthy hemisphere.

## Hypotheses, stated before the experiments (reported either way)

- **H1** (G1): how much the healthy side can help differs strongly between areas, and can be predicted from real properties of each area. *Pass:* the AI's AUC for "reaches the goal" on **held-out areas** beats the wiring-only baseline, and R² for "help" is above 0.3.
- **H2** (G2): the model's personal targets differ from today's standard target (calming the healthy M1, as in NICHE/E-FIT) in most patients and do better. Calming the healthy M1 brings no benefit, which reproduces NICHE. *Pass:* the personal target beats the standard target in >50% of simulated patients, and the standard target is within ±1 point of practice alone.
- **H3** (G3): practice without moving, combined with personal targets, reaches the goal with at least 50% fewer physical repetitions than standard care in most areas where the goal is reachable.
- **H4** (G4): the best time to start differs between areas and damage levels.
- **H7** (G7): there is a level of disease spread at which the callosum turns from helper into highway. This is tested in a later phase.

## Honest limits

- The patients are simulated.
- The wiring is a group average.
- Several therapy strengths are literature-based assumptions (docs/factors.md).
- Results are testable predictions, not clinical proof.
