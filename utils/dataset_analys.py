from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
fid.compute()










# >>> import torch
# >>> _ = torch.manual_seed(123)
# >>> from torchmetrics.image.inception import InceptionScore
# >>> inception = InceptionScore()
# >>> # generate some images
# >>> imgs = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8)
# >>> inception.update(imgs)
# >>> inception.compute()
#







from typing import Any, List, Optional, Sequence, Union, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from typing_extensions import Literal

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

_DEFAULT_MODEL: str = "openai/clip-vit-large-patch14"

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor


    def _download_clip() -> None:
        _CLIPModel.from_pretrained(_DEFAULT_MODEL)
        _CLIPProcessor.from_pretrained(_DEFAULT_MODEL)


    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_clip):
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]


class CLIPIScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0

    score: Tensor
    n_samples: Tensor
    plot_upper_bound = 100.0

    def __init__(
            self,
            model_name_or_path: Literal[
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
            ] = _DEFAULT_MODEL,  # type: ignore[assignment]
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _clip_score_update(
            images1: Union[Image.Image, List[Image.Image]],
            images2: Union[Image.Image, List[Image.Image]],
            model: _CLIPModel,
            processor: _CLIPProcessor,
    ) -> Tuple[Tensor, int]:
        if len(images1) != len(images2):
            raise ValueError(
                f"Expected the number of images to be the same but got {len(images1)} and {len(images2)}"
            )

        device = next(model.parameters()).device
        img1_processed_input = processor(images=images1, return_tensors="pt")
        img2_processed_input = processor(images=images2, return_tensors="pt")

        img1_features = model.get_image_features(img1_processed_input["pixel_values"].to(device))
        img1_features = img1_features / img1_features.norm(p=2, dim=-1, keepdim=True)

        img2_features = model.get_image_features(img2_processed_input["pixel_values"].to(device))
        img2_features = img2_features / img2_features.norm(p=2, dim=-1, keepdim=True)

        score = 100 * (img1_features * img2_features).sum(axis=-1)
        return score, len(images1)

    def update(self, images1: Union[Image.Image, List[Image.Image]],
               images2: Union[Image.Image, List[Image.Image]]) -> None:
        score, n_samples = self._clip_score_update(images1, images2, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))












