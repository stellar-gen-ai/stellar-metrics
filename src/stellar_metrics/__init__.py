__version__ = "0.0.1"


from stellar_metrics.attribute_preservation import AttributePreservation
from stellar_metrics.clip_based import CLIPMetrics
from stellar_metrics.dino_subject_fidelity import DINOFidelity
from stellar_metrics.dreamsim_distance import DreamsimDistance
from stellar_metrics.human_preference_score import HumanPreferenceScore
from stellar_metrics.identity_preservation import IdentityPreservation
from stellar_metrics.identity_stability import IdentityStability
from stellar_metrics.image_reward import ImageReward
from stellar_metrics.models.aesthetic_predictor import AestheticPredictor
from stellar_metrics.object_faithfulness import ObjectFaithfulness
from stellar_metrics.pick_score import PickScore

METRICS = {
    "ips": ("identity_preservation", IdentityPreservation),
    "goa": ("object_faithfulness", ObjectFaithfulness),
    "aps": ("attribute_preservation", AttributePreservation),
    "clip": ("clip", CLIPMetrics),
    "aesth": ("aesthetic_score", AestheticPredictor),
    "pick": ("pick_score", PickScore),
    "im_reward": ("image_reward", ImageReward),
    "hps": ("human_preference_score", HumanPreferenceScore),
    "dino": ("dino", DINOFidelity),
    "dreamsim": ("dreamsim", DreamsimDistance),
    "sis": ("identity_stability", IdentityStability),
}

metric_names = list(METRICS.keys())

from stellar_metrics.__main__ import run as run_metric
