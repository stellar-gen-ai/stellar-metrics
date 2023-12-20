import torch
from PIL import Image

from stellar_metrics.models import OwlViT
from stellar_metrics.utils import ld_to_dl

OBJECTS2SKIP = [
    "clown",
    "businessman",
    "doctor",
    "nurse",
    "cop",
    "policeman",
    "firefighter",
    "fireman",
    "soldier",
    "captain",
    "admiral",
    "scientist",
    "knight",
    "DJ",
    "egyptian pharaoh",
    "king",
    "emperor",
    "astronaut",
    "cowboy",
    "wizard",
    "pilot",
    "Pope",
    "President",
    "sumo fighter",
]


class ObjectFaithfulness:
    def __init__(self, device, dtype) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.object_detector = OwlViT(device=device, dtype=dtype)

    def prepare_detectables(self, detectables: list[list[str]]):
        lengths = torch.tensor(
            [len(detectables_signel) for detectables_signel in detectables],
            device=self.device,
        )
        detectables_ids = torch.concat([
            torch.tensor(
                [idx for idx, _ in enumerate(detectables_signel)],
                device=self.device,
            )
            for detectables_signel in detectables
        ]).unsqueeze(1)

        return detectables_ids, lengths

    @torch.no_grad()
    def __call__(
        self,
        syn_img: list[Image.Image],
        detectables: list[list[str]] = [],
        return_predictions: bool = False,
    ):
        detectables = [
            [label for label in detectables_signel if label not in OBJECTS2SKIP]
            for detectables_signel in detectables
        ]
        output = torch.full([len(syn_img)], -1.0, device=self.device)
        if detectables == []:
            if return_predictions:
                return output.cpu().numpy(), None
            return output.cpu().numpy()

        # inference
        predictions = ld_to_dl(self.object_detector(syn_img, detectables))

        # wanted labels from string to int
        detectables, reduce_lengths = self.prepare_detectables(detectables)
        # Object Faithfulness Score
        # create one "score row" per wanted label for this image
        predictions_scores_rep = torch.repeat_interleave(
            predictions["scores"], reduce_lengths, dim=0
        )
        # create one "ids row" per wanted label for this image
        labels = torch.repeat_interleave(predictions["labels"], reduce_lengths, dim=0)
        # return the max score for each wanted label per image
        label_scores = torch.where(
            labels == detectables, predictions_scores_rep, 0
        ).max(dim=1)[0]
        # TODO: maxN instead of max and be able to chose more than one, e.g.
        # for multiple "person" labels etc

        # the indexes over which to apply the average reduction (remember we
        # applied repeat before, so now we average the repeated scores)
        reduce_indxs = torch.repeat_interleave(
            torch.arange(len(syn_img), device=self.device), reduce_lengths
        )

        # perform the average reduction according to reduce_indxs
        output = torch.zeros(len(syn_img), device=self.device, dtype=label_scores.dtype)
        output.scatter_reduce_(
            0, reduce_indxs, label_scores, reduce="mean", include_self=False
        )
        # now we must have a single average score per image in the range [0, 1]
        # the higher the better
        if not return_predictions:
            return output.cpu().numpy()

        return output.cpu().numpy(), predictions
