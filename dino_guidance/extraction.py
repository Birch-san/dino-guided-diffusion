from k_diffusion.evaluation import DINOv2FeatureExtractor
from torch.nn import functional as F
import torch
from torch import FloatTensor, cat
from torch.cuda.amp import autocast

class DINOv2RegFeatureExtractor(DINOv2FeatureExtractor):

  @classmethod
  def available_models(cls):
    nominal = ['vits14', 'vitb14', 'vitl14', 'vitg14']
    return [*nominal, *[f'{name}_reg' for name in nominal]]
  
  def forward(self, x: FloatTensor, shift_from_plusminus1_to_0_1=True) -> FloatTensor:
    """
    Args:
      x `FloatTensor` [batch, channels, height, width]
      shift_from_plusminus1_to_0_1 `bool`
        True indicates your tensor has domain [-1., 1.]
        False indicates your tensor has domain [0., 1.]
    """
    if shift_from_plusminus1_to_0_1:
      x = (x + 1) / 2
    x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
    if x.shape[1] == 1:
      x = cat([x] * 3, dim=1)
    x = self.normalize(x)
    with autocast(dtype=torch.float16):
      x = self.model(x).float()
    x = F.normalize(x) * x.shape[-1] ** 0.5
    return x