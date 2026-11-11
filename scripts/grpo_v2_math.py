import torch


def group_advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError('rewards must not be empty')
    centered = values - values.mean()
    std = centered.square().mean().sqrt()
    return centered / (std + 1e-6)


def token_grpo_loss(new_logp, old_logp, ref_logp, advantage, clip, beta):
    old_logp = old_logp.detach()
    ref_logp = ref_logp.detach()
    ratio = (new_logp - old_logp).exp()
    delta = ref_logp - new_logp
    kl = (delta.exp() - delta - 1).mean()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    return -objective + beta * kl, kl, (ratio - 1).abs().mean()
