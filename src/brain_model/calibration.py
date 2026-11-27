"""Published FORMULAS that turn research results into model settings (docs/calibration.md).

Each function is one equation from a paper. `scripts/calibrate.py` fits the model's settings so that
the simulated patients follow these equations, using only numbers printed in the papers
(configs/calibration/targets.yaml). Nothing here reads or changes data files.

  recovery_curve      van der Vliet et al. 2020, Ann Neurol 87:383 (412 patients): mixture model of
                      arm recovery   FM(t) = FM0 + r_k * (66 - FM0) * (1 - exp(-t / tau_k))
  subgroup_weights    the same paper's baseline distribution (logistic normal per subgroup), used to say
                      which recovery subgroup a patient with a given starting score most likely belongs to
  expected_recovery   share of the possible recovery reached after t weeks, averaged over subgroups
  rate_for_days       an exponential approach that is 95% complete after `days` (Kokinovic & Medini 2018)
  ihi_to_brake        interhemispheric inhibition ratio (Ferbert 1992; Murase 2004) -> the model's brake k
  stdp_kernel         spike-timing-dependent plasticity window (Bi & Poo 1998)
  pi_front_speed      proliferation-invasion tumour front speed v = 2 sqrt(D rho) (Swanson 2000)
  epileptor_seizes    2D Epileptor (Jirsa 2014; Proix 2014): does an area with excitability x0 seize?
  relapse_rate        relapses as a Poisson process whose rate falls with disease duration
"""
import math

import numpy as np

FM_MAX = 66.0   # Fugl-Meyer upper extremity maximum


def recovery_curve(fm0: float, t_weeks, r: float, tau: float) -> np.ndarray:
    """van der Vliet 2020, eq. 1: FM-UE after t weeks for a patient starting at fm0."""
    t = np.asarray(t_weeks, dtype=float)
    return fm0 + r * (FM_MAX - fm0) * (1.0 - np.exp(-t / tau))


def subgroup_weights(fm0: float, groups: list[dict]) -> np.ndarray:
    """P(subgroup k | starting score). The paper's baseline prior is fm0 = 66 / (1 + exp(-a)),
    a ~ Normal(mu_k, sigma_k), with group shares p_k; Bayes' rule gives the weights (the Jacobian of
    the logistic is the same for every k, so it cancels)."""
    f = min(max(fm0 / FM_MAX, 1e-3), 1 - 1e-3)
    a = math.log(f / (1 - f))
    w = np.array([g["p"] * math.exp(-0.5 * ((a - g["mu"]) / g["sigma"]) ** 2) / g["sigma"] for g in groups])
    return w / w.sum()


def expected_recovery(fm0: float, t_weeks, groups: list[dict]) -> np.ndarray:
    """Expected share of the POSSIBLE recovery (66 - fm0) reached after t weeks:
    sum_k P(k | fm0) * r_k * (1 - exp(-t / tau_k))."""
    t = np.asarray(t_weeks, dtype=float)
    w = subgroup_weights(fm0, groups)
    return sum(wk * g["r"] * (1.0 - np.exp(-t / g["tau"])) for wk, g in zip(w, groups))


def rate_for_days(days: float, steps_per_day: int, share: float = 0.95) -> float:
    """Per-step rate of x += rate * (target - x) that reaches `share` of the target after `days`:
    (1 - rate)^n = 1 - share  ->  rate = 1 - (1 - share)^(1/n)."""
    n = days * steps_per_day
    return 1.0 - (1.0 - share) ** (1.0 / n)


def ihi_to_brake(ihi_ratio: float, sender_activity: float) -> float:
    """Interhemispheric inhibition is measured as IHI = MEP(conditioned) / MEP(test alone) (Ferbert 1992).
    The model's brake keeps 1 / (1 + k * a) of an area's output, a = sender activity, so
    1 / (1 + k a) = IHI  ->  k = (1 / IHI - 1) / a."""
    if not 0 < ihi_ratio <= 1:
        raise ValueError("an inhibition ratio must be in (0, 1]")
    if sender_activity <= 0:
        raise ValueError("sender activity must be > 0")
    return (1.0 / ihi_ratio - 1.0) / sender_activity


def stdp_kernel(dt_ms: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float) -> float:
    """Bi & Poo 1998 window: weight change for one pre/post pair, dt = t_post - t_pre.
    Positive dt (sender first) strengthens: a_plus * exp(-dt / tau_plus);
    negative dt (receiver first) weakens: -a_minus * exp(dt / tau_minus)."""
    if dt_ms > 0:
        return a_plus * math.exp(-dt_ms / tau_plus)
    if dt_ms < 0:
        return -a_minus * math.exp(dt_ms / tau_minus)
    return 0.0


def pi_front_speed(d_mm2_per_day: float, rho_per_day: float) -> float:
    """Swanson proliferation-invasion model dc/dt = div(D grad c) + rho c (1 - c): the tumour edge
    moves at v = 2 sqrt(D rho) (mm/day for D in mm^2/day)."""
    return 2.0 * math.sqrt(d_mm2_per_day * rho_per_day)


def pi_network_rates(d_mm2_per_day: float, rho_per_day: float, mean_distance_mm: float,
                     steps_per_day: int) -> tuple[float, float]:
    """The same model on the 360-area wiring, dc = transport (W c - c) + growth c (1 - c) per step:
    diffusion across one inter-area distance L takes ~L^2 / D days, so transport = D / L^2 per day."""
    transport = d_mm2_per_day / mean_distance_mm ** 2 / steps_per_day
    growth = rho_per_day / steps_per_day
    return transport, growth


def epileptor_seizes(x0: float, steps: int = 40000, dt: float = 0.01, tau0: float = 2857.0,
                     current: float = 3.1) -> bool:
    """2D Epileptor (Proix et al. 2014, from Jirsa et al. 2014):
        dx/dt = 1 - x^3 - 2 x^2 - z + I
        dz/dt = (4 (x - x0) - z) / tau0
    Above a critical excitability x0 the slow variable z cannot hold x on the resting branch and the
    area keeps switching into seizures; below it, it settles at rest. Returns whether x ever reaches
    the seizure branch (x > 0) after the start-up."""
    x, z = -1.6, 3.0
    seized = False
    for k in range(steps):
        dx = 1.0 - x ** 3 - 2.0 * x ** 2 - z + current
        dz = (4.0 * (x - x0) - z) / tau0
        x += dt * dx
        z += dt * dz
        if k > steps // 4 and x > 0:
            seized = True
            break
    return seized


def epileptor_threshold(current: float = 3.1) -> float:
    """The critical excitability x0c of the 2D Epileptor: the resting point sits on the left knee of the
    fast nullcline (x = -4/3) when 4 (x - x0) = z_knee, z_knee = 1 - x^3 - 2 x^2 + I at x = -4/3.
    With I = 3.1 this is about -2.06 (the published x0c ~ -2.05). Areas with x0 above it seize."""
    xk = -4.0 / 3.0
    zk = 1.0 - xk ** 3 - 2.0 * xk ** 2 + current
    return xk - zk / 4.0


def relapse_rate(rate0: float, decline_per_year: float, years: float) -> float:
    """Relapses per year after `years` with the disease: lambda(t) = lambda0 * exp(-beta t)."""
    return rate0 * math.exp(-decline_per_year * years)
