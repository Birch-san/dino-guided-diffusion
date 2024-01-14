import torch.nn.functional as F
from torch import FloatTensor, arcsin, linalg

def spherical_dist_loss(x: FloatTensor, y: FloatTensor) -> FloatTensor:
  x = F.normalize(x, dim=-1)
  y = F.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

# by Katherine Crowson
# https://github.com/crowsonkb/clip-guided-diffusion/blob/734b068e5ece5da13bef57b3b2c2d6bea575c8a1/clip_guided_diffusion/main.py#L151C1-L153C38
def dist(u: FloatTensor, v: FloatTensor, keepdim=False) -> FloatTensor:
  norm = linalg.norm(u - v, dim=-1, keepdim=keepdim)
  return 2 * arcsin(norm / 2)