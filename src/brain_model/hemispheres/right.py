"""Right hemisphere: holistic / spatial processing.

Real-brain mode (configs/hcp.yaml): its jobs are visuospatial attention, face recognition and
emotion in the voice (prosody), all right-dominant in most people, plus vision and movement for
the LEFT half of the body and visual field.

Stand-in mode: one job, smooth waves sweeping across the whole hemisphere at once, like taking in
an entire picture rather than one piece at a time.
"""
from .base import Hemisphere


class RightHemisphere(Hemisphere):
    name = "right"
    side = 1
    default_pattern = "wave"
