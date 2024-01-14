from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
from torch import FloatTensor, Generator, randn
from torchvision.io import read_image
from os import makedirs, listdir
from os.path import join
from pathlib import Path
import fnmatch
from typing import List, Callable
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m_sde, sample_dpmpp_2m
from PIL import Image

from dino_guidance.schedule.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from dino_guidance.device import DeviceType, get_device_type
from dino_guidance.schedule.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from dino_guidance.clip_embed.embed_text_types import Embed, EmbeddingAndMask
from dino_guidance.clip_embed.embed_text import ClipImplementation, get_embedder
from dino_guidance.denoisers.unet_2d_wrapper import EPSDenoiser, VDenoiser
from dino_guidance.latents_shape import LatentsShape
# from dino_guidance.denoisers.dino_guided_nocfg_denoiser import DinoGuidedNoCFGDenoiser
# from dino_guidance.denoisers.dino_guided_cfg_denoiser import ImgBindGuidedCFGDenoiser
from dino_guidance.denoisers.nocfg_denoiser import NoCFGDenoiser
# from dino_guidance.denoisers.cfg_denoiser import CFGDenoiser
from dino_guidance.extraction import DINOv2RegFeatureExtractor
from dino_guidance.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from dino_guidance.log_intermediates import LogIntermediates, LogIntermediatesFactory, make_log_intermediates_factory
from dino_guidance.approx_vae.latents_to_pils import make_approx_latents_to_pils
from dino_guidance.approx_vae.decoder_ckpt import DecoderCkpt
from dino_guidance.approx_vae.encoder_ckpt import EncoderCkpt
from dino_guidance.approx_vae.decoder import Decoder
from dino_guidance.approx_vae.encoder import Encoder
from dino_guidance.approx_vae.get_approx_decoder import get_approx_decoder
from dino_guidance.approx_vae.get_approx_encoder import get_approx_encoder
from dino_guidance.approx_vae.latent_roundtrip import LatentsToRGB, RGBToLatents, make_approx_latents_to_rgb, make_approx_rgb_to_latents, make_real_latents_to_rgb, make_real_rgb_to_latents
from dino_guidance.approx_vae.ckpt_picker import get_approx_decoder_ckpt, get_approx_encoder_ckpt
from dino_guidance.sampling import sample_dpm_guided
from dino_guidance.spherical_dist_loss import dist, spherical_dist_loss

# relative to current working directory, i.e. repository root of embedding-compare
assets_dir = 'assets'

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

unet_dtype=torch.float16
vae_dtype=torch.float16
text_encoder_dtype=torch.float16
# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype=torch.float32

torch.set_float32_matmul_precision("high")

# WD1.5's Unet objective was parameterized on v-prediction
# https://twitter.com/RiversHaveWings/status/1578193039423852544
needs_vparam=True

# variant=None
# variant='ink'
# variant='mofu'
variant='radiance'
# variant='illusion'
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  'Birchlabs/wd-1-5-beta3-unofficial',
  torch_dtype=torch.float16,
  subfolder='unet',
  variant=variant,
).to(device).eval().requires_grad_(False)
torch.compile(unet, mode='reduce-overhead')
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  # WD1.5 uses WD1.4's VAE
  'hakurei/waifu-diffusion',
  subfolder='vae',
  torch_dtype=torch.float16,
).to(device).eval().requires_grad_(False)
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)

guidance_use_approx_vae = False
approx_decoder_ckpt: DecoderCkpt = get_approx_decoder_ckpt('waifu-diffusion/wd-1-5-beta3')
approx_decoder: Decoder = get_approx_decoder(approx_decoder_ckpt, device)
approx_encoder_ckpt: EncoderCkpt = get_approx_encoder_ckpt('waifu-diffusion/wd-1-5-beta3')
approx_encoder: Encoder = get_approx_encoder(approx_encoder_ckpt, device)
approx_latents_to_pils: LatentsToPils = make_approx_latents_to_pils(approx_decoder)
approx_guidance_decoder: LatentsToRGB = make_approx_latents_to_rgb(approx_decoder)
approx_guidance_encoder: RGBToLatents = make_approx_rgb_to_latents(approx_encoder)
real_guidance_decoder: LatentsToRGB = make_real_latents_to_rgb(vae)
real_guidance_encoder: RGBToLatents = make_real_rgb_to_latents(vae)
guidance_decoder: LatentsToRGB = approx_guidance_decoder if guidance_use_approx_vae else real_guidance_decoder
guidance_encoder: RGBToLatents = approx_guidance_encoder if guidance_use_approx_vae else real_guidance_encoder

log_intermediates_approx_decode = True
intermediate_latents_to_pils: LatentsToPils = approx_latents_to_pils if log_intermediates_approx_decode else latents_to_pils
make_log_intermediates: LogIntermediatesFactory = make_log_intermediates_factory(intermediate_latents_to_pils)
log_intermediates_enabled = False

if log_intermediates_enabled and not log_intermediates_approx_decode or not guidance_use_approx_vae:
  # if we're expecting to invoke VAE every sampler step: make it cheaper to do so
  torch.compile(vae, mode='reduce-overhead')

embed: Embed = get_embedder(
  impl=ClipImplementation.HF,
  ckpt='Birchlabs/wd-1-5-beta3-unofficial',
  variant=variant,
  # WD1.5 is conditioned on penultimate hidden state of CLIP text encoder
  subtract_hidden_state_layers=1,
  # WD1.5 trained against concatenated CLIP segments, I think usually 3 of them?
  max_context_segments=3,
  device=device,
  torch_dtype=text_encoder_dtype,
)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = VDenoiser(unet, alphas_cumprod, sampling_dtype) if needs_vparam else EPSDenoiser(unet, alphas_cumprod, sampling_dtype)

schedule_template = KarrasScheduleTemplate.Mastering
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=unet_k_wrapped.sigma_min,
  model_sigma_max=unet_k_wrapped.sigma_max,
  device=unet_k_wrapped.sigmas.device,
  dtype=unet_k_wrapped.sigmas.dtype,
)

steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: FloatTensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max.cpu(),
  sigma_min=sigma_min.cpu(),
  rho=rho,
  device=device,
).to(sampling_dtype)
# quantize
sigmas = torch.cat([
  quantize_to(sigmas[:-1], unet_k_wrapped.sigmas),
  sigmas.new_zeros(1),
])
# you could start at a later sigma if you had init noise (i.e. img2img). here we use the quantized sigma_max
start_sigma=sigmas[0]

# WD1.5 was trained on area=896**2 and no side longer than 1152
sqrt_area=896
# height = 1024
height = 896
width = sqrt_area**2//height

latent_scale_factor = 1 << (len(vae.config.block_out_channels) - 1) # in other words, 8
latents_shape = LatentsShape(unet.in_channels, height // latent_scale_factor, width // latent_scale_factor)

seed = 1234
generator = Generator(device='cpu')

noised_latents = randn((1, latents_shape.channels, latents_shape.height, latents_shape.width), dtype=sampling_dtype, device='cpu', generator=generator).to(device)
noised_latents *= start_sigma

cond = '1girl, masterpiece, extremely detailed, light smile, best quality, best aesthetic, floating hair, full body, ribbon, looking at viewer, hair between eyes, watercolor (medium), traditional media'
neg_cond = 'lowres, bad anatomy, bad hands, missing fingers, extra fingers, blurry, mutation, deformed face, ugly, bad proportions, monster, cropped, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic, realistic, real life, instagram'
# conds = [neg_cond, cond]
# cond = ''
conds = [cond]
embed_and_mask: EmbeddingAndMask = embed(conds)
embedding, _ = embed_and_mask

noise_sampler = BrownianTreeNoiseSampler(
  noised_latents,
  sigma_min=sigma_min,
  sigma_max=start_sigma,
  # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
  # I'm just re-using it because it's a convenient arbitrary number
  seed=seed,
  # transform=lambda sigma: unet_k_wrapped.sigma_to_t(sigma)
)

dino = DINOv2RegFeatureExtractor('vitl14_reg', device=device)
# torch.compile(imgbind, mode='reduce-overhead')
img_path: str = join(assets_dir, 'polka-bicubresize256-crop224-translate.png')
# img: Image.Image = Image.open(img_path)
img: FloatTensor = read_image(img_path).to(device=device, dtype=torch.float16).div_(255).unsqueeze_(0)

target_emb: FloatTensor = dino(img, shift_from_plusminus1_to_0_1=False).squeeze_(0)
guidance_scale=300.
# cfg_scale=2.0
# denoiser = ImgBindGuidedCFGDenoiser(
#   denoiser=unet_k_wrapped,
#   imgbind=imgbind,
#   latents_to_rgb=guidance_decoder,
#   target_imgbind_cond=target_imgbind_cond,
#   cross_attention_conds=embedding,
#   guidance_scale=guidance_scale,
#   cfg_scale=cfg_scale,
# )
# denoiser = DinoGuidedNoCFGDenoiser(
#   denoiser=unet_k_wrapped,
#   dino=dinov2_vitl14_reg,
#   latents_to_rgb=guidance_decoder,
#   target_emb=target_emb,
#   cross_attention_conds=embedding,
#   guidance_scale=guidance_scale,
# )
# denoiser = CFGDenoiser(
#   denoiser=unet_k_wrapped,
#   cross_attention_conds=embedding,
#   cfg_scale=7.5,
# )
denoiser = NoCFGDenoiser(
  denoiser=unet_k_wrapped,
  cross_attention_conds=embedding,
)
size_fac = (height * width) / (512 * 512)

def cond_model(x: FloatTensor, sigma: FloatTensor, **kwargs) -> FloatTensor:
  assert not x.isnan().any().item()
  denoised = None

  def loss_fn(x: FloatTensor) -> FloatTensor:
    nonlocal denoised
    denoised = denoiser(x, sigma, **kwargs)
    assert not denoised.isnan().any().item()
    denoised_rgb: FloatTensor = guidance_decoder(denoised)
    assert not denoised_rgb.isnan().any().item()
    loss = x.new_tensor(0.0)
    for target, scale in zip([target_emb], [guidance_scale]):
      image_embed = dino(denoised_rgb).squeeze_(0)
      # loss_cur = dist(image_embed, target) ** 2 / 2
      loss_cur = spherical_dist_loss(image_embed, target)
      assert not loss_cur.isnan().any().item()
      loss += loss_cur * scale * size_fac
    return loss

  grad = torch.autograd.functional.vjp(loss_fn, x)[1]
  # we don't clamp latents; thresholding remains an area of research
  # return denoised.clamp(-1, 1), -grad
  assert not grad.isnan().any().item()
  return denoised, -grad

out_dir = 'out'
makedirs(out_dir, exist_ok=True)
intermediates_dir=join(out_dir, 'intermediates')
makedirs(intermediates_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0
out_stem: str = f'{next_ix:05d}_{seed}_{guidance_scale}'

if log_intermediates_enabled:
  intermediates_path = join(intermediates_dir, out_stem)
  makedirs(intermediates_path, exist_ok=True)
  callback: LogIntermediates = make_log_intermediates([intermediates_path])
else:
  callback = None

# the maximum step size
max_h=0.1
# the maximum amount that guidance is allowed to perturb a step
max_cond=0.05
# multiplier for noise variance. 0 gives ODE sampling, 1 gives stadnard diffusion SDE sampling.
eta=1.
denoised_latents: FloatTensor = sample_dpm_guided(
  model=cond_model,
  x=noised_latents,
  sigma_min=sigma_min,
  sigma_max=start_sigma,
  max_h=max_h,
  max_cond=max_cond,
  eta=eta,
  noise_sampler=noise_sampler,
  solver_type='dpm3',
  callback=callback,
).to(vae_dtype)
del noised_latents
pil_images: List[Image.Image] = latents_to_pils(denoised_latents)
del denoised_latents


for stem, image in zip([out_stem], pil_images):
  # TODO: put this back to png once we've stopped prototyping
  out_name: str = join(out_dir, f'{stem}.jpg')
  image.save(out_name)
  print(f'Saved image: {out_name}')