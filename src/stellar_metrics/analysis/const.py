METRIC_RENAME = {
    "method": "Method",
    "clip_t": "CLIP$_T$",
    "clip_i": "CLIP$_I$",
    "aesthetic_score": "Aesth.",
    "dino_fidelity": "DINO",
    "pick_score": "PickScore",
    "dreamsim_distance": "DreamSim",
    "human_preference_score_v1": "HPSv1",
    "human_preference_score_v2": "HPSv2",
    "image_reward": "ImageReward",
    "identity_stability": "SIS",  # identity variance score
    "penet": "PENET",  # more metrics?
    "identity_preservation": "IPS",
    "attr": "APS",
    "attribute_preservation": "APS",
    "object_faithfulness": "GOA",
    "human": "HumanPreference",
    "fid": "FID",
    "relation_faithfulness_score": "RFS",
}
PERS_METRICS = [
    "object_faithfulness",
    "relation_faithfulness_score",
    "attribute_preservation",
    "identity_preservation",
    "identity_stability",
]
OBJECT_METRICS = ["object_faithfulness", "relation_faithfulness_score"]
IDENTITY_METRICS = [
    "identity_stability",
    "identity_preservation",
    "attribute_preservation",
]
I2I_METRICS = ["aesthetic_score", "clip_i", "dreamsim_distance"]
T2I_METRICS = [
    "clip_t",
    "human_preference_score_v1",
    "human_preference_score_v2",
    "image_reward",
    "pick_score",
]
METRIC_DIRECTION = {
    "clip_t": True,
    "clip_i": True,
    "aesthetic_score": True,
    "identity_preservation": True,
    "attr": True,
    "object_faithfulness": True,
    "dino_fidelity": True,
    "pick_score": True,
    "dreamsim_distance": False,
    "human_preference_score_v1": True,
    "human_preference_score_v2": True,
    "penet": True,
    "image_reward": True,
    "identity_stability": True,
}
METRIC_FILE_NAMES = [
    "attribute",
    "object",
    "aesthetic",
    "clip",
    "dino",
    "dreamsim",
    "human_preference",
    "identity_preservation",
    "identity_stability",
    "image_reward",
    "pick_score",
    "relation",
]
METHODS = {
    "stellar-net": "StellarNet (Ours)",
    "dreambooth": "DreamBooth~\\cite{dreambooth}",
    "textual_inversion": "Text. Inv. \\cite{textual_inversion}",
    "elite-celeb": "ELITE*  \\cite{elite}",
}
METHODS_AMT = {
    "stellar_net": "StellarNet (Ours)",
    "dreambooth": "DreamBooth",
    "textual_inversion": "Text. Inv.",
    "elite": "ELITE*",
}

METHOD_DISPLAY_ORDER = [
    METHODS[c]
    for c in ["dreambooth", "elite-celeb", "textual_inversion", "stellar-net"]
]
