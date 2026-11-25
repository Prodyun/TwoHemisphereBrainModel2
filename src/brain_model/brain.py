"""Assembles the two hemispheres, the corpus callosum, (optionally) a lesion and the optional
patient-care parts (therapy, stimulation, recovery) into one model."""
import numpy as np

from . import anatomy
from .connectivity import CorpusCallosum
from .connectivity.bridge_therapy import BridgeTherapy
from .connectivity.callosal_injury import CallosalInjury
from .hemispheres import LeftHemisphere, RightHemisphere
from .lesion import Lesion
from .recovery import Recovery
from .therapy import Stimulation, Therapy


def steps_per_day(config: dict) -> int:
    return int(config.get("time", {}).get("steps_per_day", 8))


class Brain:
    def __init__(self, config: dict, with_lesion: bool = True):
        rng = np.random.default_rng(config["seed"])
        n = config["n_regions"]
        dyn = config["dynamics"]
        anat = config.get("anatomy")
        self.left = LeftHemisphere(n, config["left_hemisphere"], dyn, rng, anat)
        self.right = RightHemisphere(n, config["right_hemisphere"], dyn, rng, anat)
        cc = config["corpus_callosum"]
        into_left, into_right = anatomy.callosal_wiring(anat, cc.get("wiring", "mirror"), n)
        self.callosum = CorpusCallosum(cc["strength"], n, cc.get("plasticity_rate", 0.0),
                                       cc.get("plasticity_gain", 0.0), cc.get("takeover_gain", 0.0),
                                       into_left, into_right, cc.get("inhibition", 0.0))
        self.lesion = None
        self.therapy = self.stimulation = self.recovery = self.bridge = self.callosal_injury = None
        self.trained = None
        if with_lesion and config.get("lesion"):
            target_name = config["lesion"]["hemisphere"]
            if target_name not in ("left", "right"):
                raise ValueError("lesion.hemisphere must be 'left' or 'right'")
            cc_ = self.callosum
            if target_name == "left":
                pair, routes = (self.left, self.right), (cc_.route_into_left, cc_.route_into_right)
            else:
                pair, routes = (self.right, self.left), (cc_.route_into_right, cc_.route_into_left)
            self.lesion = Lesion({"steps_per_day": steps_per_day(config), **config["lesion"]}, *pair, *routes)
            if config.get("callosal_injury"):
                self.callosal_injury = CallosalInjury(config["callosal_injury"], self.callosum, steps_per_day(config),
                                                      self.lesion.start_step,
                                                      np.random.default_rng(config["seed"] + 13))
            self._patient_care(config, pair[0], pair[1])
        self.reps_physical = self.reps_mental = 0.0

    def _patient_care(self, config: dict, damaged, healthy) -> None:
        """Therapy, stimulation and healing (all optional; they start counting from the injury)."""
        spd = steps_per_day(config)
        onset = self.lesion.start_step
        th = config.get("therapy") or {}
        if th and th.get("mode", "none") != "none":
            side = th.get("hemisphere", "damaged")
            self.trained = damaged if side == "damaged" else (self.left if side == "left" else self.right)
            self.therapy = Therapy(th, self.trained, spd, onset, np.random.default_rng(config["seed"] + 7))
        st = config.get("stimulation") or {}
        if st.get("targets"):
            self.stimulation = Stimulation(st, self.left, self.right, spd, onset)
        rec = config.get("recovery") or {}
        if rec.get("enabled", False):
            self.recovery = Recovery(rec, len(self.left.x), spd, onset)
        br = config.get("bridge") or {}
        if br.get("mode") and br.get("pairs"):
            self.bridge = BridgeTherapy(br, self.callosum, damaged, healthy, spd, onset,
                                        np.random.default_rng(config["seed"] + 11))

    def step(self, t: int, noise_left: np.ndarray, noise_right: np.ndarray) -> dict:
        """One time step: exchange signals, update both hemispheres, then spread damage, heal and adapt."""
        parts_left, parts_right = self.left.task_parts(t), self.right.task_parts(t)
        task_left, task_right = sum(parts_left.values()), sum(parts_right.values())
        practice = None
        if self.therapy:
            parts = parts_left if self.trained is self.left else parts_right
            healthy = self.right if self.trained is self.left else self.left
            practice = self.therapy.step(t, parts[self.therapy.function], healthy.x)
            if self.trained is self.left:
                task_left = task_left + practice.input
            else:
                task_right = task_right + practice.input
        practising = practice is not None and practice.kind == 1
        stim = {s: (self.stimulation.gain(s, t, practising) if self.stimulation else 1.0)
                for s in ("left", "right")}
        sent = {s: (self.stimulation.output(s, t, practising) if self.stimulation else 1.0)
                for s in ("left", "right")}
        into_left, into_right = self.callosum.signals(self.left, self.right, sent["left"], sent["right"])
        brake_left, brake_right = self.callosum.brakes(self.left, self.right)
        cc = self.callosum
        extra = {"left": None, "right": None}
        if self.bridge:  # Mirror Drive: the healthy side's activity triggers stimulation of the damaged side
            extra[self.bridge.damaged.name] = self.bridge.drive(
                t, practising, practice.pattern if practice is not None else None)
        self.left.step(t, into_left, noise_left, partner_task=cc.takeover_by_left * task_right, task=task_left,
                       own_scale=brake_left * stim["left"], extra=extra["left"])
        self.right.step(t, into_right, noise_right, partner_task=cc.takeover_by_right * task_left, task=task_right,
                        own_scale=brake_right * stim["right"], extra=extra["right"])
        if self.lesion:
            self.lesion.update(t)
        if self.callosal_injury:
            self.callosal_injury.update(t)
        if self.recovery:
            for hemi in (self.left, self.right):
                use = None
                if practice is not None and practice.kind and hemi is self.trained:
                    # Hebbian use: activity that moves TOGETHER with the practised pattern drives
                    # repair, plus the matching input arriving from the healthy side (the callosal
                    # synapses carrying the practised movement are strengthened too)
                    arriving = into_left if hemi is self.left else into_right
                    use = practice.use_scale * (np.maximum(hemi.x * practice.pattern, 0.0)
                                                + self.recovery.callosal_use_weight
                                                * np.maximum(arriving * practice.pattern, 0.0))
                self.recovery.update(t, hemi, use, practice.boost if use is not None else 1.0,
                                     practice.session_reps if practice is not None else 0.0)
        self.callosum.adapt(self.left, self.right)
        if self.bridge:  # Bridge Builder: paired pulses strengthen the chosen fibres
            self.bridge.update(t, practising)
        if practice is not None:
            self.reps_physical += practice.physical_reps
            self.reps_mental += practice.mental_reps
        out = {
            "left": self.left.x.copy(), "right": self.right.x.copy(),
            "left_damage": self.left.damage.copy(), "right_damage": self.right.damage.copy(),
            "into_left": into_left, "into_right": into_right,
            # link strengths used this step (they change with plasticity); the viewer uses them to
            # show the signal on each callosal fibre
            "gain_into_left": cc.gain_into_left.copy(), "gain_into_right": cc.gain_into_right.copy(),
            "left_task": task_left, "right_task": task_right,
        }
        if self.therapy or self.stimulation:
            out["practice"] = practice.kind if practice is not None else 0
            out["stimulating"] = bool(self.stimulation and self.stimulation.active(t, practising))
            out["bci_fired"] = bool(practice is not None and practice.bci_fired)
            out["reps_physical"] = self.reps_physical
            out["reps_mental"] = self.reps_mental
        if self.lesion is not None and self.lesion.kind == "epilepsy":
            out["seizure"] = self.lesion.seizing
        if self.callosal_injury:
            out["callosal_integrity"] = self.callosal_injury.mean_integrity()
        if self.bridge:
            out["bridge_fired"] = self.bridge.last_fired
            out["bridge_weights"] = self.bridge.pair_weights()
        for side, parts in (("left", parts_left), ("right", parts_right)):
            for name, value in parts.items():
                out[f"{side}_task:{name}"] = value
        return out
