# What already exists, and the gap this project fills

Collected on 2026-11-24 from web searches. Each item still needs checking against the paper before it goes
on the poster: search summaries can be wrong.

| Step | What exists today | What it does not do |
|---|---|---|
| Detect | FDA-cleared stroke AI: RapidAI, Viz.ai and Brainomix (vessel occlusion, core and penumbra on CT). DeepISLES (Nature Communications 2025, doi 10.1038/s41467-025-62373-x), an open MRI lesion segmenter that matches experts. EEG Brain Symmetry Index for detecting stroke before hospital: delta-band AUC 0.91 (ELECTRA-STROKE); subhairline EEG AUC 0.90 (2023). | Stops at "where and how big". Nothing about recovery or the healthy side. |
| Spread | Connectome models predict tau spread from each person's own epicentres (Franzmeier et al. 2020, Science Advances; fully individualised models, 2025). Network diffusion models for AD, FTD, ALS and PD (Raj et al. 2012, Neuron). | Research only, and not linked to a treatment choice. |
| Prognosis | PREP2 (Stinear et al. 2017, Ann Clin Transl Neurol) predicts arm outcome from strength, TMS and NIHSS; about 75% accurate; used in hospitals. | Says *how much* recovery, not *what to do*. |
| Where to stimulate | One-size targets. NICHE (Harvey 2018) and E-FIT (Stroke 2023): 1 Hz rTMS calming the healthy M1 was not better than sham. The bimodal balance-recovery model (Di Pino et al. 2014, Nat Rev Neurol) says the right choice depends on each patient's reserve. Personalised rTMS stroke trials are at protocol stage (NCT06270238, NCT06951035, NCT07049211). In depression, personalised targeting is FDA-cleared (SAINT, 2022). | No tool picks the healthy-side target per stroke patient. |
| How much | SimNIBS electric-field models: lesions change the field by up to 47%, and standard tDCS may underdose stroke patients. | Field strength, not the recovery it buys. |
| Therapies | FDA approved: Vivistim paired VNS (2021; VNS-REHAB, Dawson et al. 2021 Lancet). FDA authorised: IpsiHand BCI (2021), which reads the healthy hemisphere. | Single devices. Nothing compares the options for one patient. |
| Healthy side treats the damaged side | ccPAS (paired cortico-cortical stimulation): a sham-controlled Phase II stroke trial with 44 patients found FMA-UE +6.3 vs +3.2 (Brain Communications 2024); review by Falaschi, Stagg and Sela 2025. CCFES: the healthy hand drives stimulation of the weak hand (Knutson et al. 2016, Stroke). Brain-state-triggered rTMS is feasible in stroke (Frontiers in Neurology 2024). | Nobody chooses the ccPAS pair per patient. No stroke study triggers damaged-side stimulation from healthy-side activity. |
| Digital twins | Virtual Epileptic Patient / EPINOV (356 patients, 11 centres, ends late 2026). Whole-brain models predict stroke deficits from the lesion (Idesis et al. 2024; Falcon et al. 2016). | Plan epilepsy surgery or explain deficits; no two-hemisphere recovery help. |

**The gap (the ISEF pitch):** no tool connects, for one patient:

- detect;
- spread risk;
- a personal healthy-side target;
- a dose;
- ranked therapy options, each linked to a real device or trial;
- a split-brain control;
- the healthy side actively treating the damaged side (Bridge Builder, Mirror Drive, Hand-off).

## Sources (links)

- DeepISLES: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12335569/
- ELECTRA-STROKE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12296629/
- Franzmeier 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7695466
- Individualised tau models 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12690294/
- Raj 2012: https://pmc.ncbi.nlm.nih.gov/articles/PMC3623298
- PREP2: https://pmc.ncbi.nlm.nih.gov/articles/PMC5682112
- E-FIT: https://pmc.ncbi.nlm.nih.gov/articles/PMC10453351
- Bimodal model: https://www.doi.org/10.1038/NRNEUROL.2014.162
- Personalised rTMS protocol: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1427142/full
- ccPAS Phase II: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11653076/
- ccPAS review 2025: https://repository.essex.ac.uk/41625/
- CCFES: https://pmc.ncbi.nlm.nih.gov/articles/PMC5039083
- Brain-state rTMS in stroke: https://frontiersin.org/articles/10.3389/fneur.2024.1427198/full
- Vivistim approval: https://www.neurologylive.com/view/fda-approves-vivistim-vns-system-ischemic-stroke-motor-deficits
- IpsiHand: https://source.washu.edu/2021/04/stroke-recovery-device-using-brain-computer-interface-receives-fda-market-authorization/
- SAINT: https://clinicaltmssociety.org/ctmss-press-release-regarding-fda-clearance-of-saint-neuromodulation-system/
- SimNIBS and lesions: https://pmc.ncbi.nlm.nih.gov/articles/PMC5426045
- EPINOV: https://ebrains.eu/news-and-events/2025/epinov-clinical-trial-virtual-brain-technology-to-support-surgery-in-drug
- Idesis whole-brain stroke model: https://pmc.ncbi.nlm.nih.gov/articles/PMC11285191
- ATLAS v2.0 (not downloaded; the owner decided to keep data unchanged): https://pmc.ncbi.nlm.nih.gov/articles/PMC9203460/
- Lesion network mapping debate: https://www.thetransmitter.org/brain-imaging/overdue-debate-unfurls-over-neuroimaging-method/
