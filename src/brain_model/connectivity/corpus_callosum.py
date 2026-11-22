"""The corpus callosum: the bridge that lets the two hemispheres share information."""
from ..hemispheres import Hemisphere


class CorpusCallosum:
    def __init__(self, strength: float):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("corpus_callosum.strength must be between 0 and 1")
        self.strength = strength

    def exchange(self, left: Hemisphere, right: Hemisphere) -> None:
        """strength 0 = no sharing (split brain); 1 = fully merged (both hemispheres identical)."""
        left_out, right_out = left.state.copy(), right.state.copy()
        mix = self.strength / 2  # at mix 0.5 each side becomes an equal blend of both
        left.receive(right_out, mix)
        right.receive(left_out, mix)
