# How it works: the concept in plain language

## The idea in one picture

```
   DAMAGE                     HELP                          REPAIR                      THE AI
 an area on one side   the healthy side sends       practice + matching help     predicts, for ANY area:
 is hurt (stroke,  ->  matching signals across  ->  "fire together, wire     ->  • chance the help works
 tumour, ...)          the corpus callosum          together" (Hebbian)          • which areas carry it
                                                    -> the area heals            • best practice mode
                                                                                 • how many repetitions
```

**Your brain model is two hemispheres made from real wiring.** That wiring comes from the Human Connectome Project's brain scans, with 180 areas per side. When an area is damaged, the other hemisphere keeps sending signals across the corpus callosum. In the model, the damaged area heals when it is **used**, and it counts as used when its activity moves **together with** the practised movement. The matching signals from the healthy side count too. So the healthy side does more than cover for the damage: it helps **repair** it.

## What happens when you click an area

1. **Identify.** The clicked area plus its 4 most strongly wired neighbours become a "patient". If the area has no job in the model (many don't), it gets a local job so its recovery can be measured.
2. **The AI predicts.** In under a second, the AI Help Predictor gives:
   - the chance that the other side can help this area reach 80% of normal
   - the expected help, in points against a split brain
   - which practice mode is best
   - **why**: which properties of this area matter
3. **The tracer explains how.** It measures how much matching help each healthy-side area actually sends. For example, "15% from R_4 by a direct fibre, 14% from R_6d…", or "61% from L_RSC in two steps, via R_PH" for areas with no direct fibre.
4. **Simulate.** The viewer runs 4 months of care for three plans:
   - the AI's plan
   - standard care (physical practice)
   - the AI's plan with the callosum cut (the control)
   
   You can add **your own plan** ("Beat the AI").
5. **Watch it happen:**
   - the day counter runs
   - practice sessions light up
   - for BCI: *"imagining → decoder detected 'trying' → feedback ✓"*
   - VNS pulses
   - ⚡ stimulation flashing on the helper areas
   - gold arcs carrying help
   - repetitions counting up, physical and mental
   - the damage fading
   - the weekly chart climbing toward the goal

## What you see in the viewer

- **The damage develops over time.** Pick one of three disorders:
  - **Stroke:** the core is hit at once. The at-risk ring around it (the *penumbra*) dies off over the first 5 days, and the banner says so.
  - **Tumour:** grows slowly and presses on its neighbours.
  - **Spreading disease:** starts in the clicked area and travels along the brain's wiring for months. Newly reached areas flash, and the wiring carrying the disease glows orange.
- **The help travels, and you can see it:**
  - **Separate view:** gold pathways connect each healthy-side helper area (e.g. R_6a) to the damaged area it reaches (L_6a). They are labelled with each helper's share, and comets flow along them. They follow each brain as you rotate it.
  - **Together view:** comets travel along gold arcs over the corpus callosum, and the damaged area glows while help arrives.
  - **Split brain:** the pathways turn into red dashed lines with "callosum cut: no help".
- Brightness and speed follow the real help signal in the simulation at that moment.

## The AI model

- **Type:** gradient-boosted decision trees (scikit-learn), a standard, explainable machine-learning method.
- **Training data:** about 7,200 simulated patients:
  - every one of the 360 areas damaged
  - 2 severities, stroke and tumour
  - 5 practice modes
  - each case simulated connected and split
- **What it looks at:** real data about each area:
  - its fibres to the other side, and how strongly it syncs with its mirror area (HCP)
  - its network (Cole-Anticevic)
  - its section (Glasser 2016)
  - plasticity and disease genes (Allen Human Brain Atlas)
  - receptor chemistry (Hansen 2022 PET)
- **How it's tested:** whole areas are held out. Every score is measured on areas the AI **never saw** in training, and compared with simple baselines. See `results/help_model/report.md`.

## The live BCI loop (the next step, with an EEG cap)

1. The person *imagines* squeezing their right hand, without moving it.
2. The 8–30 Hz rhythm over the left motor cortex (electrode C3) **drops**.
3. The **decoder** (a small program) decides "trying" or "resting".
4. When it detects trying, **feedback** happens instantly. In a clinic this is an electrical muscle pulse or a robotic glove. In our demo, it is the virtual patient's practice step.
5. Because the feedback comes at the exact moment of effort, the brain strengthens the circuit (Hebbian learning). Timed feedback gave +16.2 arm-function points, against +4.8 at random times (*Sci Rep* 2024).

In the demo, each detected attempt becomes a repetition for the simulated patient. The screen states clearly that this shows the BCI input working on a real brain, **not** a real patient recovering.

## The research behind each part

| Model part | What the research found | How sure we are | Source |
|---|---|---|---|
| Golden period | Extra therapy at 2–3 months helped most (+6.87 ARAT), ≤30 days less (+5.25), ≥6 months little (+2.41, n.s.) | One trial; the rate at which the window closes was never measured | CPASS, Dromerick 2021 PNAS |
| Repetitions plateau | 3,200 vs 9,600 repetitions made no difference in chronic stroke | Phase II RCT | Lang 2016 Ann Neurol |
| VNS doubles gains | +5.0 vs +2.4 Fugl-Meyer; no effect unless paired with movement | Pivotal RCT plus rat studies | Dawson 2021 Lancet; Hays 2014 |
| BCI timing | Feedback timed to effort beats random timing | Small double-blind RCT | Sci Rep 2024 |
| Imagery | Uses premotor/parietal areas more than M1 | Imaging meta-analysis; exact strength unknown | Hardwick 2018 |
| Today's target fails | Calming the healthy M1 did not beat sham | Phase 3 RCT (n=199) and E-FIT | NICHE; E-FIT 2023 |
| Healthy PMd can help | In severe patients, the healthy premotor cortex facilitates the damaged side | TMS-fMRI study | Bestmann 2010 J Neurosci |
| IpsiHand | An FDA-authorized BCI reads the healthy hemisphere | Uncontrolled study, n=40 | FDA De Novo 2021 |

## What is real, what is simulated, what is assumed

- **Real:** brain wiring, brain syncing, area maps, networks, genes, receptors, standard TMS sites, and the trial numbers used for calibration.
- **Simulated:** the patients, their recovery, and the effects of therapy.
- **Assumed** (listed in `docs/factors.md`): the time scale (1 step = 1 hour), imagery strength, VNS multiplier, BCI bonus, sleep replay strength, golden-period shape, and how much damage is permanent.

## What to say to the judges

- "Our model **predicts**. These are testable predictions, not clinical proof."
- "Every score of the AI is on brain areas it never saw during training."
- "We checked the model against published trials it was not tuned to."
- "To our knowledge, no published work combines a per-area AI, help pathways, personal targets and repetition-free practice in one model (see docs/novelty.md)."
- Limitations: group-average wiring, simulated patients, assumed therapy strengths, cortex only (no basal ganglia).

## Glossary

| Term | Meaning |
|---|---|
| Corpus callosum | The bundle of fibres that connects the two hemispheres |
| Homotopic | Mirror areas, the same area on the left and on the right |
| Hebbian | "Neurons that fire together wire together" |
| BCI | Brain-computer interface |
| FES | Electrical stimulation that makes muscles move |
| VNS | Vagus nerve stimulation |
| TMS / rTMS | Magnetic stimulation of the brain from outside the head |
| AUC | How well the AI separates "will reach the goal" from "won't" (0.5 = chance, 1 = perfect) |
| R² | How much of the variation the AI explains (0 = nothing, 1 = everything) |
