from typing import Any, List, Optional, Sequence, Union, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Module as _DINOModel
from torchmetrics import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import transforms
from torchvision.transforms import Compose as _DINOProcessor
from typing_extensions import Literal

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DINOScore.plot"]

_DEFAULT_MODEL: str = "dino_vits16"


class DINOScore(Metric):

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
                "dino_vits16",
            ] = _DEFAULT_MODEL,  # type: ignore[assignment]
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = self._get_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _get_model_and_processor(
            model_name_or_path: Literal[
                "dino_vits16",
            ] = "dino_vits16",
    ) -> Tuple[_DINOModel, _DINOProcessor]:
        if _TRANSFORMERS_AVAILABLE:
            model = torch.hub.load('facebookresearch/dino:main', model_name_or_path)
            processor = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            return model, processor

        raise ModuleNotFoundError(
            "`dino_score` metric requires `transformers` package be installed."
            " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[multimodal]`."
        )

    @staticmethod
    def _dino_score_update(
            images1: Union[Image.Image, List[Image.Image]],
            images2: Union[Image.Image, List[Image.Image]],
            model: _DINOModel,
            processor: _DINOProcessor,
    ) -> Tuple[Tensor, int]:
        if len(images1) != len(images2):
            raise ValueError(
                f"Expected the number of images to be the same but got {len(images1)} and {len(images2)}"
            )

        device = next(model.parameters()).device

        img1_processed_input = [processor(i) for i in images1]
        img2_processed_input = [processor(i) for i in images2]

        img1_processed_input = torch.stack(img1_processed_input).to(device)
        img2_processed_input = torch.stack(img2_processed_input).to(device)

        img1_features = model(img1_processed_input)
        img2_features = model(img2_processed_input)

        # cosine similarity between feature vectors
        score = 100 * F.cosine_similarity(img1_features, img2_features, dim=-1)
        return score, len(images1)

    def update(self, images1: Union[Image.Image, List[Image.Image]],
               images2: Union[Image.Image, List[Image.Image]]) -> None:
        score, n_samples = self._dino_score_update(images1, images2, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated dino score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)