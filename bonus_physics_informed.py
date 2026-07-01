"""
BONUS: physics-informed constraint (mentors explicitly called this out for
bonus scoring: "design and implement physics informed modeling for the task").

Physical assumption encoded here (document this exact sentence in your
report): thermal infrared radiance is a monotonic proxy for surface
brightness temperature (Stefan-Boltzmann: radiant exitance grows with T^4,
and satellite-measured TIR digital numbers are a scene-scaled,
monotonically increasing function of it). So a *hotter* pixel in the raw
TIR band should not become a *visually darker/cooler-looking* pixel in the
colorized RGB output, and vice versa -- the colorization should not invert
the physical brightness-temperature ordering it was conditioned on.

This is implemented as a soft monotonicity/rank-consistency penalty
between the input TIR intensity and the perceptual luminance of the
generated RGB output, computed at the patch level (cheap, no extra model,
~1-2 hours to implement and integrate into train_colorize.py).

To use: add this loss term into train_colorize.py's generator loss with a
small weight (e.g. 1.0-3.0) alongside the existing L1 / adversarial /
semantic-constraint terms.
"""
import torch
import torch.nn as nn


class RadiativeMonotonicityLoss(nn.Module):
    """
    Penalizes rank-order disagreement between input TIR brightness and
    output RGB luminance, computed per local patch (not globally, since
    absolute color depends on land-cover class, not just temperature --
    only the *local relative ordering* is physically constrained).
    """

    def __init__(self, patch: int = 16):
        super().__init__()
        self.patch = patch
        # standard Rec. 601 luma weights
        self.register_buffer("luma_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        return (rgb * self.luma_weights).sum(dim=1, keepdim=True)

    def _patchify_mean(self, x: torch.Tensor):
        b, c, h, w = x.shape
        p = self.patch
        x = x[:, :, : h - h % p, : w - w % p]
        x = x.unfold(2, p, p).unfold(3, p, p)
        return x.mean(dim=(-1, -2))  # (B, C, H//p, W//p)

    def forward(self, tir_input: torch.Tensor, generated_rgb: torch.Tensor):
        luminance = self._luminance(generated_rgb)
        tir_patch_mean = self._patchify_mean(tir_input).flatten(1)          # (B, N)
        lum_patch_mean = self._patchify_mean(luminance).flatten(1)          # (B, N)

        # Soft rank-consistency via pairwise-sign agreement, sampled (not
        # full O(N^2) — subsample pairs for speed on larger patch grids)
        n = tir_patch_mean.shape[1]
        if n < 2:
            return torch.tensor(0.0, device=tir_input.device)
        idx_a = torch.randperm(n, device=tir_input.device)[: max(2, n // 2)]
        idx_b = torch.randperm(n, device=tir_input.device)[: max(2, n // 2)]

        tir_diff = tir_patch_mean[:, idx_a] - tir_patch_mean[:, idx_b]
        lum_diff = lum_patch_mean[:, idx_a] - lum_patch_mean[:, idx_b]

        # Penalize cases where sign(tir_diff) != sign(lum_diff), scaled by
        # magnitude of the TIR difference (small differences are noise,
        # not a physical claim worth enforcing)
        disagreement = torch.relu(-tir_diff * lum_diff)
        return disagreement.mean()


if __name__ == "__main__":
    loss_fn = RadiativeMonotonicityLoss()
    tir = torch.randn(2, 1, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)
    val = loss_fn(tir, rgb)
    print("RadiativeMonotonicityLoss OK:", val.item())
