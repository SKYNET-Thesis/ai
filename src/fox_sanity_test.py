#!/usr/bin/env python
"""
Safety tests for the FoX patch. Run BEFORE training (after fox_patch.py).

    python fox_sanity_test.py            # math tests only (CPU, seconds)
    python fox_sanity_test.py --model    # + model-level tests on lerobot/smolvla_base (needs a GPU)

Math tests      : paper formula, zero diagonal, causality.
Model-level     : (1) alpha=0 reproduces the baseline exactly (train loss and sampled actions)
                  (2) full uncached forward == prefill + cached denoise_step with alpha=1 / random gates
                  (3) alpha receives a gradient at init, gates are finite (fp32), no NaN
"""
import argparse
import math

import torch

from lerobot.policies.smolvla.smolvlm_with_expert import ForgetGate

MODEL_ID = "lerobot/smolvla_base"


# ------------------------------------------------------------------------------ math
def test_math():
    torch.manual_seed(0)
    B, H, L, D, E = 2, 3, 12, 16, 24
    gate = ForgetGate(E, H, bias_init=0.0, identity_init=False)  # alpha = 1
    nn_init = torch.nn.init.normal_
    nn_init(gate.proj.weight, std=0.5)
    x = torch.randn(B, L, E)
    q, k, v = (torch.randn(B, H, L, D) for _ in range(3))

    bias = gate(x)  # [B,H,L,L]
    assert torch.allclose(torch.diagonal(bias, dim1=-2, dim2=-1), torch.zeros(B, H, L)), "diagonal must be 0"

    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    s = q @ k.transpose(-1, -2) / math.sqrt(D) + bias
    p = torch.softmax(s.masked_fill(~causal, torch.finfo(s.dtype).min), dim=-1)
    out_bias = p @ v

    # explicit paper formula: o_i = sum_j F_ij exp(q_i.k_j) v_j / sum_j F_ij exp(q_i.k_j), F_ij = prod_{l=j+1..i} f_l
    f = torch.sigmoid(gate.proj(x)).transpose(1, 2)  # [B,H,L]
    out_ref = torch.zeros_like(out_bias)
    for i in range(L):
        w = []
        for j in range(i + 1):
            Fij = f[:, :, j + 1 : i + 1].prod(-1) if j < i else torch.ones(B, H)
            w.append(Fij * torch.exp((q[:, :, i] * k[:, :, j]).sum(-1) / math.sqrt(D)))
        w = torch.stack(w, -1)  # [B,H,i+1]
        out_ref[:, :, i] = (w[..., None] * v[:, :, : i + 1]).sum(-2) / w.sum(-1, keepdim=True)
    assert torch.allclose(out_bias, out_ref, atol=1e-5), (out_bias - out_ref).abs().max()

    # causality: changing input token t+1.. must not change bias rows <= t (lower triangle)
    t = 5
    x2 = x.clone()
    x2[:, t + 1 :] += torch.randn_like(x2[:, t + 1 :])
    b2 = gate(x2)
    assert torch.allclose(bias[:, :, : t + 1, : t + 1], b2[:, :, : t + 1, : t + 1], atol=1e-6)

    # identity init
    g0 = ForgetGate(E, H, identity_init=True)
    assert gate_is_zero(g0(x))
    print("[ok] math tests: paper formula, zero diagonal, causality, identity init")


def gate_is_zero(b):
    return bool((b == 0).all())


# ------------------------------------------------------------------------------ model-level
def fake_batch(policy, B=2, device="cuda"):
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    cfg = policy.config
    batch = {}
    for key, ft in cfg.image_features.items():
        c, h, w = ft.shape
        batch[key] = torch.rand(B, c, h, w, device=device)
    batch[OBS_STATE] = torch.randn(B, cfg.robot_state_feature.shape[0], device=device)
    batch[ACTION] = torch.randn(B, cfg.chunk_size, cfg.action_feature.shape[0], device=device)
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (B, 16), device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(B, 16, dtype=torch.bool, device=device)
    return batch


def load_policy(use_fox, device="cuda"):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(MODEL_ID)
    cfg.device = device
    cfg.use_fox = use_fox
    cfg.fox_identity_init = True
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=cfg).to(device)
    return policy


def unpack(policy, batch):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state, actions


@torch.no_grad()
def uncached_vt(vf, images, img_masks, lang, lm, state, x_t, time):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    pe, ppm, pam = vf.embed_prefix(images, img_masks, lang, lm, state=state)
    se, spm, sam = vf.embed_suffix(x_t, time)
    pad = torch.cat([ppm, spm], 1)
    att = torch.cat([pam, sam], 1)
    (_, suf), _ = vf.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, att),
        position_ids=torch.cumsum(pad, 1) - 1,
        past_key_values=None,
        inputs_embeds=[pe, se],
        use_cache=False,
    )
    return vf.action_out_proj(suf[:, -vf.config.chunk_size :].float())


@torch.no_grad()
def cached_vt(vf, images, img_masks, lang, lm, state, x_t, time):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    pe, ppm, pam = vf.embed_prefix(images, img_masks, lang, lm, state=state)
    _, pkv = vf.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(ppm, pam),
        position_ids=torch.cumsum(ppm, 1) - 1,
        past_key_values=None,
        inputs_embeds=[pe, None],
        use_cache=True,
    )
    return vf.denoise_step(prefix_pad_masks=ppm, past_key_values=pkv, x_t=x_t, timestep=time)


def test_model():
    from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs

    dev = "cuda"
    base = load_policy(False, dev).eval()
    fox = load_policy(True, dev).eval()
    # make both models identical outside the gates (fox was loaded from the same checkpoint)
    miss = [k for k in fox.state_dict() if k not in base.state_dict()]
    assert miss and all("fox_gates" in k for k in miss), f"unexpected extra keys: {miss[:5]}"
    print(f"[info] {len(miss)} new tensors, all fox_gates.*")

    torch.manual_seed(0)
    batch = fake_batch(base, device=dev)
    B = batch[next(iter(base.config.image_features))].shape[0]
    noise = torch.randn(B, base.config.chunk_size, base.config.max_action_dim, device=dev)
    time = torch.rand(B, device=dev) * 0.9 + 0.05

    ia = unpack(base, dict(batch))
    ib = unpack(fox, dict(batch))

    # (1) alpha = 0  ->  baseline, exactly
    with torch.no_grad():
        la = base.model.forward(*ia, noise, time)
        lb = fox.model.forward(*ib, noise, time)
        assert torch.allclose(la, lb, atol=1e-6), (la - lb).abs().max()
        sa = base.model.sample_actions(*ia[:5], noise=noise)
        sb = fox.model.sample_actions(*ib[:5], noise=noise)
        assert torch.allclose(sa, sb, atol=1e-6), (sa - sb).abs().max()
    print("[ok] (1) alpha=0 reproduces the baseline (loss + sampled actions)")

    # (3) alpha gets gradient at init, proj gets zero grad (alpha=0), everything finite
    fox.train()
    fox.zero_grad()
    loss = fox.model.forward(*ib, noise, time).mean()
    loss.backward()
    a_grads = [g.alpha.grad for g in fox.model.vlm_with_expert.fox_gates.values()]
    assert all(g is not None and torch.isfinite(g).all() for g in a_grads)
    assert any(g.abs().sum() > 0 for g in a_grads), "alpha receives no gradient -> it could never leave 0"
    assert torch.isfinite(loss)
    print(f"[ok] (3) alpha grad non-zero at init, loss finite ({loss.item():.4f})")
    fox.eval()
    fox.zero_grad(set_to_none=True)

    # (2) alpha = 1, random gates: uncached training path == prefill + cached denoise path
    for g in fox.model.vlm_with_expert.fox_gates.values():
        torch.nn.init.normal_(g.proj.weight, std=0.5)
        torch.nn.init.normal_(g.proj.bias, std=1.0)
        g.alpha.data.fill_(1.0)
    x_t, _, _ = make_flow_matching_inputs(ib[5], noise, time, convention=FlowConvention.NOISE_AT_ONE)
    v_unc = uncached_vt(fox.model, *ib[:5], x_t, time)
    v_cac = cached_vt(fox.model, *ib[:5], x_t, time)
    err = (v_unc - v_cac).abs().max().item()
    assert err < 1e-3, f"train path and cached path disagree (max abs err {err:.3e}) -> bias padding is wrong"
    # and the gate must actually change the output
    for g in fox.model.vlm_with_expert.fox_gates.values():
        g.alpha.data.zero_()
    v0 = uncached_vt(fox.model, *ib[:5], x_t, time)
    assert (v0 - v_unc).abs().max() > 1e-4, "gate has no effect"
    print(f"[ok] (2) uncached vs cached path agree (max abs err {err:.2e}); gate changes output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    args = ap.parse_args()
    test_math()
    if args.model:
        test_model()
    print("ALL TESTS PASSED")
