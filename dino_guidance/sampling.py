import torch
import k_diffusion as K

# from clip-guided-diffusion
# by Katherine Crowson
# https://github.com/crowsonkb/clip-guided-diffusion/blob/734b068e5ece5da13bef57b3b2c2d6bea575c8a1/clip_guided_diffusion/main.py#L348C1-L461C1
@torch.no_grad()
def sample_dpm_guided(
    model,
    x,
    sigma_min,
    sigma_max,
    max_h,
    max_cond,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    callback=None,
):
    """DPM-Solver++(1/2/3M) SDE (Kat's splitting version)."""
    noise_sampler = (
        K.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
        if noise_sampler is None
        else noise_sampler
    )
    if solver_type not in {"euler", "midpoint", "heun", "dpm3"}:
        raise ValueError('solver_type must be "euler", "midpoint", "heun", or "dpm3"')

    # Helper functions
    def sigma_to_t(sigma):
        return -torch.log(sigma)

    def t_to_sigma(t):
        return torch.exp(-t)

    def phi_1(h):
        return torch.expm1(-h)

    def h_for_max_cond(t, eta, cond_eps_norm, max_cond):
        # This returns the h that should be used for the given cond_scale norm to keep
        # the norm of its contribution to a step below max_cond at a given t.
        sigma = t_to_sigma(t)
        h = (cond_eps_norm / (cond_eps_norm - max_cond / sigma)).log() / (eta + 1)
        return h.nan_to_num(nan=float("inf"))

    # Set up constants
    sigma_min = torch.tensor(sigma_min, device=x.device)
    sigma_max = torch.tensor(sigma_max, device=x.device)
    max_h = torch.tensor(max_h, device=x.device)
    s_in = x.new_ones([x.shape[0]])
    t_end = sigma_to_t(sigma_min)

    # Set up state
    t = sigma_to_t(sigma_max)
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None
    i = 0

    # Main loop
    while t < t_end - 1e-5:
        # Call model and cond_fn
        sigma = t_to_sigma(t)
        denoised, cond_score = model(x, sigma * s_in)

        # Scale step size down if cond_score is too large
        cond_eps_norm = cond_score.mul(sigma).pow(2).mean().sqrt() + 1e-8
        h = h_for_max_cond(t, eta, cond_eps_norm, max_cond)
        h = max_h * torch.tanh(h / max_h)
        t_next = torch.minimum(t + h, t_end)
        h = t_next - t
        sigma_next = t_to_sigma(t_next)

        # Callback
        if callback is not None:
            obj = {
                "x": x,
                "i": i,
                "sigma": sigma,
                "sigma_next": sigma_next,
                "denoised": denoised,
            }
            callback(obj)

        # First order step (guided)
        h_eta = h + eta * h
        x = (sigma_next / sigma) * torch.exp(-h * eta) * x
        x = x - phi_1(h_eta) * (denoised + sigma**2 * cond_score)
        noise = noise_sampler(sigma, sigma_next)
        x = x + noise * sigma_next * phi_1(2 * eta * h * s_noise).neg().sqrt()

        # Higher order correction (not guided)
        if solver_type == "dpm3" and denoised_2 is not None:
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = phi_1(h_eta) / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        elif solver_type in {"heun", "dpm3"} and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = phi_1(h_eta) / h_eta + 1
            x = x + phi_2 * d
        elif solver_type == "midpoint" and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            x = x - 0.5 * phi_1(h_eta) * d

        # Update state
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
        t += h
        i += 1

    return x
