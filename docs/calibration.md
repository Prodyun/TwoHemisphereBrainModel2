# Calibration: from guessed numbers to published formulas

Many settings in `configs/care.yaml` started as guesses. This page explains how each one is now tied
to a **published formula**. The model is run on virtual patients, and the setting is chosen so that
the simulated patients follow the formula. Only numbers printed in the papers are used
(`configs/calibration/targets.yaml`); no new data was downloaded and nothing in `data/` was changed.

- Formulas: `src/brain_model/calibration.py`
- Fitting: `python scripts/calibrate.py` writes `results/calibration/report.md`
- Tests: `tests/test_calibration.py`

Every setting carries a label in `docs/factors.md`:
- **calibrated**: set by fitting a published formula;
- **supported**: in line with published evidence, but no number could be fitted;
- **assumption**: still a choice, swept and stated on the poster.

## Healing: van der Vliet 2020 and CPASS 2021

**Formula.** The van der Vliet mixture model fitted 412 stroke patients' arm scores
(FM-UE, maximum 66) week by week:

    FM(t) = FM0 + r_k (66 - FM0) (1 - exp(-t / tau_k))

There are 5 recovery subgroups, from poor (r = 0.09) to excellent (r = 0.93, tau = 1.2 weeks).
The patient's starting score FM0 tells how likely each subgroup is (Bayes' rule with the paper's
Table 1). The model's expected share of possible recovery after t weeks is therefore

    sum over k of P(k | FM0) * r_k * (1 - exp(-t / tau_k))

**CPASS.** 20 extra practice hours started at 2-3 months gave +6.87 ARAT points (of 57) more than
standard care, in patients starting at 13.4 points. That is 15.8% of their starting deficit.

**How it is fitted.** 36 virtual motor strokes (6 areas, both sides, small to large) get standard
care. For each patient:
1. Measure the starting function from a run with no healing.
2. Measure the share of possible recovery each week. It is compared with a copy of the same
   patient with healing switched off, so week-to-week timetable effects cancel.
3. Compare that share with the formula.

The extra-practice gain of the most severe third is compared with CPASS. A grid search over
`floor_fraction`, `spontaneous_per_day` and `use_rate` picks the best match.

**Why a new rule was needed.** With only the old three settings, no value matched both papers. Self-healing
and practice both repaired damage down to the same fixed floor, so practice changed how *fast* a patient got
better but never how *far*. The recovery curves need fast self-healing, so virtual patients reached the floor
in about 3 weeks, and the extra practice of CPASS at day 75 had nothing left to fix (0.2% of the deficit vs
15.8% in the trial). In CPASS, extra practice raised the final score after usual care had levelled off. So
practice may now also lower the floor:

    relief += window x floor_use_rate x boost x saturation x use x (1 - relief)
    floor   = floor_fraction x worst damage x (1 - relief)

`floor_use_rate` is fitted together with the other three settings to the recovery curves and to CPASS.
At 0 the model behaves exactly as before.

**Assumption we must state.** The model has no Fugl-Meyer scale, so its function score × 66 is read
as FM-UE. With the bridge cut, a fully destroyed movement job scores ~0 (a true zero), which
supports this reading.

## Bridge: Kokinovic & Medini 2018, Murase 2004, DCM

- **`plasticity_rate`** (calibrated). After a touch-cortex stroke in mice, responses came back
  through the callosum by day 14. A rule x += rate (target − x) that is 95% done in 14 days of
  24 steps has rate = 1 − 0.05^(1/336) = **0.00888** (it was a guess of 0.02, which finished in
  about 2 days).
- **The brake** (the interhemispheric inhibition onto movement areas).
  - The TMS measure is IHI = MEP(conditioned) / MEP(test). The model's brake keeps 1/(1 + k a), so
    **k = (1/IHI − 1)/a**. When Murase 2004's ratio is filled in, k follows from this one line.
  - Until then the brake is swept and checked against three published patterns:
    - NICHE / E-FIT: calming the healthy M1 is not better than sham;
    - Di Pino 2014: calming helps some patients with high reserve, fewer with low reserve;
    - DCM (Rehme 2011): the healthy M1's influence on the damaged M1 is supportive early and
      becomes inhibitory late in poorly recovered patients. This is measured on the model's own
      activity by regression, the way DCM measures real brains.
  - A brake value is accepted only if all three hold; otherwise it stays off.

## Healthy-side therapies

- **Bridge Builder timing (STDP, Bi & Poo 1998).** Δw = A₊ e^(−Δt/τ₊) when the healthy pulse
  comes first, −A₋ e^(Δt/τ₋) when it comes second (τ₊ 16.8 ms, τ₋ 33.7 ms).
  - Setting `bridge.pair.isi_ms` makes the strengthening depend on the gap, relative to the 8 ms
    used in ccPAS (Buch 2011).
  - A negative gap weakens the link: the reverse-order control now follows from the formula.
- **Mirror Drive gain.** To be fitted to the CCFES trial effect (Knutson 2016) when its number is
  filled in.

## Diseases (options; the old behaviour stays the default until numbers are filled in)

| Disease | Formula | Setting |
|---|---|---|
| Tumour | Proliferation-invasion (Swanson): dc/dt = div(D grad c) + ρ c (1 − c), front speed 2√(Dρ); on the wiring, transport = D/L², growth = ρ | `lesion.model: proliferation_invasion`, `rho_per_day` 0.012, `d_mm2_per_day` 0.13 |
| Epilepsy | 2D Epileptor (Jirsa 2014): an area seizes only if its excitability x0 is above x0c = −4/3 − z_knee/4 ≈ −2.06 | `lesion.focus_x0` |
| MS | Relapses as a Poisson process with a falling rate λ(t) = λ0 e^(−βt) (Tremlett 2008) | `callosal_injury.relapse_decline_per_year` |
| Stroke | Penumbra lost exponentially, 95% by `penumbra_days` | `lesion.penumbra_shape: exponential` |

## Numbers still to fill in from full texts

These are not blockers; the model runs without them.
- Murase 2004: IHI ratio.
- Paul 2023: callosal r.
- Buch 2011: MEP gain after one session.
- Knutson 2016: effect size.
- Tremlett 2008: relapse decline.

## Results (2026-11-27, 36 virtual patients)

| Setting | Was (guess) | Now | How well it is pinned down |
|---|---|---|---|
| `spontaneous_per_day` | 0.003 | 0.16 | calibrated (recovery curves) |
| `floor_fraction` | 0.3 | 0.3 | calibrated |
| `floor_use_rate` (new) | – | 0.03 | calibrated, but CPASS not matched |
| `use_rate` | 0.03 | 0.3 | weak: 0.03 fits almost as well |
| `plasticity_rate` | 0.02 | 0.00888 | calibrated (Kokinovic & Medini) |
| brake `inhibition` | 0 | 0 | no value passed all three checks: stays off |

- **Recovery curves:** matched well (mean squared error 0.004 of the recovered share).
- **CPASS:** not matched. Extra practice at 2-3 months removes 1.6% of the deficit (trial 15.8%). The virtual
  "severe" patients still start at about 30-37 of 66 points and recover to about 90%, so too little deficit
  is left for extra practice to fix. The trial enrolled weaker arms (ARAT 13 of 57).
- **Brake:** at every value, calming the healthy side was not better than sham (NICHE ✔). Calming was
  almost never the best choice for any patient (Di Pino ✖), so the brake stays off.
- **Main result, connected vs split (12 strokes, 4 motor areas, 3 severities):** the connected brain had
  less leftover damage in every case (−0.9 points on average) and +0.8 points more function on average.
  With the old guessed healing it was +6.5 points. In left-side strokes the split brain scored slightly
  higher function. **Why:** with realistic healing speed, mild-to-moderate virtual strokes mostly heal on their own, so
  the healthy side has little left to add. Its help matters most when a job is destroyed, where the split
  brain stays near zero.
- **Trial checks:** NICHE/E-FIT ✔; VNS-REHAB ½ (function +1.6 points, responders not higher);
  CPASS timing ✖.
- The AI Help Predictor was trained under the old settings and has not been retrained (data rule).
