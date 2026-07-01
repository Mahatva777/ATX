"""
Semantic-constraint loss for colorization.

The PS explicitly requires "Preserve Semantic Integrity" and the mentor
transcript flags hallucination as a scored qualitative risk. Two options,
pick based on remaining time budget:

OPTION A (fast, ~30 min to implement, recommended for a 30-hour hackathon):
    Band-ratio consistency loss. No extra model needed. Uses the physical
    fact that water absorbs strongly in NIR/SWIR and vegetation reflects
    strongly in NIR, so simple band-ratio indices computed on the *TIR
    input* should correlate with color/brightness patterns in the
    *generated RGB output*. We approximate this without a NIR band (not
    available here) by enforcing that pixels with distinct TIR signatures
    (thermal outliers -- e.g. water bodies run cooler and more uniform
    than urban/bare soil at similar times of day) map to low color
    variance in the corresponding generator output patch. This is a
    proxy, not a true spectral index -- document it as such.

OPTION B (higher fidelity, needs a pretrained/frozen classifier):
    A frozen land-cover classifier (e.g. a small CNN pretrained on
    EuroSAT or a similar land-cover dataset) scores the *real* RGB and
    the *generated* RGB; a cross-entropy / KL term penalizes the
    generator when its output's predicted land-cover class disagrees
    with the ground truth's class. This is closer to "semantic
    integrity" in the literal sense but costs GPU time to train/fine-tune
    and adds a dependency on an external pretrained model + label set,
    which may not be feasible inside 30 hours.

Both are provided below. train_colorize.py defaults to Option A and can
be switched to Option B if a classifier is available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BandRatioConsistencyLoss(nn.Module):
    """
    OPTION A. Cheap, dependency-free semantic-consistency proxy.

    Idea: locally uniform / cold TIR regions (candidate water bodies)
    should produce locally low-variance, blue-shifted color in the
    generator's output; locally warm & highly-textured TIR regions
    (candidate urban/built-up) should produce higher local color variance.
    We do NOT claim this replaces a real land-cover classifier -- it is a
    lightweight structural prior that penalizes obviously inconsistent
    outputs (e.g. a smooth, cold TIR patch coming out mottled bright red).
    """

    def __init__(self, patch: int = 8, water_temp_percentile: float = 0.15):
        super().__init__()
        self.patch = patch
        self.water_temp_percentile = water_temp_percentile

    @staticmethod
    def _local_stats(x: torch.Tensor, patch: int):
        # x: (B, C, H, W) -> local mean/var per non-overlapping patch
        b, c, h, w = x.shape
        x = x[:, :, : h - h % patch, : w - w % patch]
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)  # (B,C,H//p,W//p,p,p)
        mean = x.mean(dim=(-1, -2))
        var = x.var(dim=(-1, -2))
        return mean, var

    def forward(self, tir_input: torch.Tensor, generated_rgb: torch.Tensor):
        tir_mean, tir_var = self._local_stats(tir_input, self.patch)  # (B,1,h',w')
        rgb_mean, rgb_var = self._local_stats(generated_rgb, self.patch)  # (B,3,h',w')
        rgb_var_avg = rgb_var.mean(dim=1, keepdim=True)  # collapse channel -> (B,1,h',w')
        blue_channel_mean = generated_rgb[:, 2:3]
        blue_mean, _ = self._local_stats(blue_channel_mean, self.patch)

        # "Cold & uniform" TIR patches -> identify via low percentile threshold per-image
        flat = tir_mean.flatten(1)
        thresh = torch.quantile(flat, self.water_temp_percentile, dim=1).view(-1, 1, 1, 1)
        water_like = (tir_mean < thresh).float()

        # Penalty 1: water-like regions should have low RGB local variance
        variance_penalty = (water_like * rgb_var_avg).mean()
        # Penalty 2: water-like regions should skew toward higher blue mean than red/green
        red_mean, _ = self._local_stats(generated_rgb[:, 0:1], self.patch)
        blue_bias_penalty = (water_like * F.relu(red_mean - blue_mean)).mean()

        return variance_penalty + blue_bias_penalty


class LandCoverClassifierLoss(nn.Module):
    """
    OPTION B. Requires a frozen, pretrained land-cover classifier
    `classifier(rgb_batch) -> logits`. Pass one in; this module does not
    define or train the classifier itself (out of scope for the hackathon
    time budget unless a suitable pretrained checkpoint is available,
    e.g. a small ResNet fine-tuned on EuroSAT-RGB).
    """

    def __init__(self, frozen_classifier: nn.Module):
        super().__init__()
        self.classifier = frozen_classifier
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

    def forward(self, generated_rgb: torch.Tensor, real_rgb: torch.Tensor):
        with torch.no_grad():
            target_logits = self.classifier(real_rgb)
            target_probs = F.softmax(target_logits, dim=1)
        pred_logits = self.classifier(generated_rgb)
        pred_log_probs = F.log_softmax(pred_logits, dim=1)
        return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")


if __name__ == "__main__":
    loss_fn = BandRatioConsistencyLoss()
    tir = torch.randn(2, 1, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)
    val = loss_fn(tir, rgb)
    print("BandRatioConsistencyLoss OK:", val.item())
