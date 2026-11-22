# Project Requirements

Fill in what you know. Leave anything blank if you're unsure, and Claude will ask.
Plain language is fine.

## 1. Goal
What is this project for? (research, school/class project, demo, learning, paper...)

> Science fair project, aiming for ISEF level. Build a simulation of the brain as two distinct
> hemisphere models (left and right). A disorder starts at one point in one hemisphere and spreads.
> The question: can the other hemisphere send signals across (through the corpus callosum) to
> help the damaged hemisphere keep working? Not by overriding it, but by collaborating, because
> the two hemispheres are distinct yet share some qualities.

## 2. What should the LEFT hemisphere do?
(e.g. language, reading text, logic, math, sequences, speech)

> Language and math/sequences (left-dominant), plus vision and movement for the right side of the
> body. Each job drives its real Glasser areas (see `configs/hcp.yaml`).

## 3. What should the RIGHT hemisphere do?
(e.g. images, shapes, spatial reasoning, faces, emotion, music, big-picture patterns)

> Spatial attention, faces and emotion in the voice (prosody) (right-dominant), plus vision and
> movement for the left side of the body.

## 4. How should the two hemispheres interact?
(e.g. share everything, share a little, compete, one checks the other, "split-brain" experiments)

> Distinct models that share some qualities: each has its own real wiring, settings and jobs, but the
> same population equation. Never mirror copies of each other. When one hemisphere is damaged, the healthy one
> sends signals across the corpus callosum to support it. Compare against a split brain
> (callosum cut) as the control.

## 5. What kind of AI?
- [ ] Small neural networks trained on tasks
- [x] Biologically inspired neuron (spiking) models. Chosen: a brain-region network model
      (each region is a simplified population of neurons), to be built from HCP brain-scan data
- [ ] Two large language model agents (e.g. Claude) with different roles that talk to each other
- [ ] Not sure, recommend one

## 6. What should I be able to see or do at the end?
(e.g. charts, a web page, a chat interface, a report, numbers comparing hemispheres)

> A 3D simulation on a **black background** (so the brain is easy to see) showing **two separate
> models**, a left and a right hemisphere, that can each be **rotated separately**. Show the
> disorder spreading and the signals coming from the healthy side.

## 7. Experiments you'd like to run
(e.g. "what happens when the connection is cut?", "does the left side learn language faster?")

- Connected brain vs split brain: how much does the healthy hemisphere help?
- Disorder in the left vs in the right hemisphere
- Does the callosum also spread the disorder (`lesion.cross_spread`)? Trade-off: helping vs. spreading
- With vs without plasticity (the brain adapting after damage)
- Spreading disorder vs stroke (sudden damage that does not spread)
- Real measured callosum fibres vs mirror-only links vs both
- Which abilities (language, vision, ...) does the healthy side rescue?

## 8. Constraints
- Deadline: finish the build within 2-3 days of 2026-11-21
- Computer (does it have a graphics card, GPU?): laptop, no GPU
- Budget for paid AI services (if any):

## 9. Anything else / inspiration
(papers, videos, books, ideas)

> Data: Human Connectome Project Young Adult (HCP-YA), 3T Preprocessed Structural, Diffusion and
> Resting-State fMRI, subjects with "Full 3T imaging protocol = True", no 7T data.
> Pilot: 2 subjects (resting fMRI only, to check the model). The wiring, atlas and brain shape come
> from HCP group data via the ENIGMA Toolbox (no login). See docs/data.md.
