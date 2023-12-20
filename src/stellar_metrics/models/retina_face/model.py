import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDector:
    def __init__(self, device: str | None = None, margin=0.9) -> None:
        self.mtcnn = MTCNN(margin=margin, device=device, keep_all=True).eval()
        # If required, create a face detection pipeline using MTCNN:

        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

    @torch.no_grad()
    def __call__(self, img) -> None | torch.Tensor:
        # Get cropped and prewhitened image tensor
        img_cropped = self.mtcnn(img)
        # Calculate embedding (unsqueeze to add batch dimension)
        if img_cropped is None:
            return None
        img_embedding = self.resnet(img_cropped)
        return img_embedding
