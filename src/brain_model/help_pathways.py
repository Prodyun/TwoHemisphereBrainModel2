"""HOW does the healthy hemisphere help? Trace the help that reaches the damaged areas, per sender.

During practice, damaged area i receives from each healthy area j (through the corpus callosum):

    help(j -> i) = gain_into_i x route[i, j] x  mean over practice steps of  max(0, sent_j x pattern_i)

where sent_j is j's activity (scaled by its health) and pattern_i is the practised movement at i.
Only activity that moves WITH the practised movement counts (the same Hebbian rule the healing
uses). Each healthy area's share of the total help is reported, with the kind of route:

  direct fibre   a fibre measured by diffusion MRI between j and the damaged area
  mirror link    j is the damaged area's matching area and no fibre was measured (tractography misses
                 many callosal fibres; the model adds a mirror link when `wiring: both` or `mirror`)
and, for the plain-language explanation, the healthy-side neighbour through which a sender with
no direct fibre still reaches the damage (two steps: neighbour -> sender -> across).
"""
import numpy as np

from .brain import Brain


def trace(cfg: dict, run: dict, top: int = 8) -> dict:
    side = cfg["lesion"]["hemisphere"]
    other = "right" if side == "left" else "left"
    brain = Brain(cfg, with_lesion=True)
    damaged = brain.left if side == "left" else brain.right
    healthy = brain.right if side == "left" else brain.left
    cc = brain.callosum
    route = cc.into_left_wiring if side == "left" else cc.into_right_wiring
    n = len(damaged.x)
    R = np.eye(n) if route is None else route              # receiver (damaged) x sender (healthy)
    job = cfg["therapy"]["function"]
    regions = np.asarray(run["functions"][side][job])
    practice = np.asarray(run.get("practice", np.zeros(len(run[side]))), int) == 1
    if not practice.any():                                 # no therapy: use every step after the injury
        practice = np.arange(len(run[side])) >= cfg["lesion"]["start_step"]
    pattern = run[f"{side}_task:{job}"][practice][:, regions]          # (T, job areas)
    health = 1.0 - run[f"{other}_damage"][practice]                    # (T, n)
    sent = health * run[other][practice]                                # (T, n) healthy side's output
    gain = run[f"gain_into_{side}"][practice][:, regions]              # (T, job areas)
    dmg = run[f"{side}_damage"][practice][:, regions].mean(axis=0)     # weight damaged receivers more
    # matched[t, a, j] = max(0, sent_j(t) x pattern_a(t)); help[a, j] = mean_t(gain x route x matched)
    matched = np.maximum(sent[:, None, :] * pattern[:, :, None], 0.0)  # (T, job areas, n)
    help_aj = (gain[:, :, None] * R[regions][None, :, :] * matched).mean(axis=0)
    # Two steps: j -> (across) -> damaged-side area k -> (own wiring) -> job area a. Many areas have
    # no measured fibre across (e.g. L_44, R_FFC); their help arrives this way.
    gain_all = run[f"gain_into_{side}"][practice].mean(axis=0)                     # (n,)
    health_d = 1.0 - run[f"{side}_damage"][practice].mean(axis=0)                  # (n,)
    via = damaged.gain * damaged.W[regions] * (health_d * gain_all)[None, :]       # (a, k)
    indirect_aj = (via @ R) * matched.mean(axis=0)                                  # (a, j)
    via_k = {}
    for j in range(n):
        contrib = (via * R[:, j][None, :]).sum(axis=0)                             # through each k
        via_k[j] = int(np.argmax(contrib)) if contrib.max() > 0 else None
    help_aj = help_aj + indirect_aj
    per_sender = (dmg[:, None] * help_aj).sum(axis=0)                  # (n,)
    total = per_sender.sum()
    share = per_sender / total if total > 0 else per_sender
    lr = None
    if cfg.get("anatomy", {}).get("source") == "hcp":
        from .hcp import load_hcp_brain
        b = load_hcp_brain(cfg["anatomy"]["file"])
        lr = b.sc_left_right if side == "left" else b.sc_left_right.T  # [damaged i, healthy j]
    helpers = []
    for j in np.argsort(share)[::-1][:top]:
        if share[j] <= 0:
            break
        receivers = regions[np.argsort(help_aj[:, j])[::-1][:2]]
        direct = lr is not None and bool((lr[regions, j] > 0).any())
        dname = (lambda i: damaged.region_names[i] if damaged.region_names else str(i))
        if direct:
            kind = "direct fibre"
        elif j in regions and route is None:
            kind = "mirror link"
        elif via_k[j] is not None:
            kind = f"two steps, via {dname(via_k[j])}"
        else:
            kind = "via the callosal wiring"
        helpers.append({"area": healthy.region_names[j] if healthy.region_names else str(j), "index": int(j),
                        "share": round(float(share[j]), 4), "route": kind,
                        "via": dname(via_k[j]) if via_k[j] is not None else None,
                        "reaches": [dname(i) for i in receivers]})
    return {"damaged_hemisphere": side, "healthy_hemisphere": other, "job": job,
            "total_help": float(total), "helpers": helpers}


def explain(t: dict) -> list[str]:
    """Plain-language sentences for the panel and reports."""
    if not t["helpers"]:
        return ["No help reaches the damaged areas: the callosum is cut or the areas have no links across."]
    lines = []
    for h in t["helpers"][:4]:
        lines.append(f"{h['share'] * 100:.0f}% of the help comes from {h['area']} ({h['route']}), "
                     f"reaching {', '.join(h['reaches'])}.")
    top = ", ".join(h["area"] for h in t["helpers"][:2])
    lines.append(f"For this to work, {top} must be active WITH the practised movement: practice that "
                 f"engages both sides, and making {top} more excitable, increase the help.")
    return lines
