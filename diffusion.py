import torch

class DDPM:
    '''
    Minimal DDPM utilities (noise schedule, q_sample, and sampling).
    '''
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = int(T)
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        '''
        x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        x0: (B,C,H,W), t: (B,)
        '''
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    @torch.no_grad()
    def sample(self, model, shape, cond=None, extra_channels=None):
        '''
        Generate samples starting from Gaussian noise on the data channel.
        If extra_channels is given (e.g., coords), it is concatenated to x at every step.

        shape: (B,1,H,W) for the data channel.
        extra_channels: (B,k,H,W) or None
        '''
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_in = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            x_in = torch.cat([x, extra_channels], dim=1) if extra_channels is not None else x
            eps = model(x_in, t_in, cond=cond)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            coef1 = beta_t * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - alpha_bar_t)
            coef2 = (1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x

            if t == 0:
                x = mean
            else:
                var = self.posterior_variance[t]
                x = mean + torch.sqrt(var) * torch.randn_like(x)
        return x
