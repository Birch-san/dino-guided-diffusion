from k_diffusion.evaluation import DINOv2FeatureExtractor

class DINOv2RegFeatureExtractor(DINOv2FeatureExtractor):
  @classmethod
  def available_models(cls):
    nominal = ['vits14', 'vitb14', 'vitl14', 'vitg14']
    return [*nominal, *[f'{name}_reg' for name in nominal]]