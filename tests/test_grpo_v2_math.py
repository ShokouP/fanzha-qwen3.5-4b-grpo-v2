import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from grpo_v2_math import group_advantages, token_grpo_loss


def test_identity_gradient_and_detached_reference():
    new = torch.tensor([-1., -2.], requires_grad=True)
    old = new.detach().clone().requires_grad_()
    ref = new.detach().clone().requires_grad_()
    loss, kl, deviation = token_grpo_loss(new, old, ref, 1., .2, .05)
    loss.backward()
    assert deviation.item() == 0 and abs(kl.item()) < 1e-6
    assert new.grad.abs().sum() > 0 and old.grad is None and ref.grad is None


def test_advantages():
    adv = group_advantages([1., 2., 3., 4.])
    assert abs(adv.mean().item()) < 1e-6
    assert torch.equal(group_advantages([2., 2.]), torch.zeros(2))


def test_kl_and_clip():
    new = torch.tensor([0.5, -0.5], requires_grad=True)
    old = torch.zeros(2)
    ref = torch.tensor([0.2, 0.2], requires_grad=True)
    loss, kl, _ = token_grpo_loss(new, old, ref, 1., .2, 0.)
    assert kl.item() >= 0
    loss.backward()
    assert new.grad[0] == 0 and new.grad[1] < 0 and ref.grad is None

if __name__ == "__main__":
    test_identity_gradient_and_detached_reference()
    test_advantages()
    test_kl_and_clip()
    print("GRPO math checks passed")
