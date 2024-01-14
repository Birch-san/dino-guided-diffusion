from .unet_2d_wrapper import Denoiser
from torch import FloatTensor, BoolTensor, enable_grad
from torch import autograd
from dataclasses import dataclass
from typing import Optional, Tuple
from ..approx_vae.latent_roundtrip import LatentsToRGB
from ..extraction import DINOv2RegFeatureExtractor
from ..spherical_dist_loss import spherical_dist_loss

@dataclass
class DinoGuidedNoCFGDenoiser:
  denoiser: Denoiser
  dino: DINOv2RegFeatureExtractor
  latents_to_rgb: LatentsToRGB
  cross_attention_conds: FloatTensor
  target_emb: FloatTensor
  guidance_scale: float = 50.
  cross_attention_mask: Optional[BoolTensor] = None

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    noised_latents = noised_latents.detach().requires_grad_()
    with enable_grad():
      denoised: FloatTensor = self.denoiser.forward(
        input=noised_latents,
        sigma=sigma,
        encoder_hidden_states=self.cross_attention_conds,
        cross_attention_mask=self.cross_attention_mask,
      )
      decoded: FloatTensor = self.latents_to_rgb(denoised)
      emb: FloatTensor = self.dino(decoded)
      loss: FloatTensor = spherical_dist_loss(emb, self.target_emb).sum() * self.guidance_scale
      vec_jacobians: Tuple[FloatTensor, ...] = autograd.grad(loss, noised_latents)
      grad: FloatTensor = -vec_jacobians[0].detach()
    guided_cond: FloatTensor = denoised.detach() + grad * sigma**2
    return guided_cond