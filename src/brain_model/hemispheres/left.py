"""Left hemisphere: sequential / analytical processing.

Real-brain mode (configs/hcp.yaml): its jobs are language and math/sequences (left-dominant in
most people), plus vision and movement for the RIGHT half of the body and visual field, which both
hemispheres do for their own opposite side.

Stand-in mode: one job, a pulse moving front -> back through all regions, like processing the
words of a sentence in order.
"""
from .base import Hemisphere


class LeftHemisphere(Hemisphere):
    name = "left"
    side = -1
    default_pattern = "sequence"
