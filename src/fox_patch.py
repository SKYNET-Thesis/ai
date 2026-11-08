#!/usr/bin/env python
"""
Patch LeRobot's SmolVLA in place so that the action expert's *self-attention*
layers use FoX (Forgetting Attention, Lin et al., ICLR 2025).

What is changed (everything else is untouched):
  * configuration_smolvla.py   : new flags use_fox / fox_bias_init / fox_identity_init / fox_alpha_lr
  * smolvlm_with_expert.py     : ForgetGate module, gate bias added to the action->action block of
                                 the self-attention layers (even layers), VLM loaded in fp32
  * modeling_smolvla.py        : flags passed through, model cast to fp32, optimizer param groups
                                 (gate alpha gets its own LR, no weight decay on gates)

Cross-attention layers (odd layers), KV cache, masks, RoPE, flow-matching loss: unchanged.

Usage:
    python fox_patch.py                       # patches the installed lerobot
    python fox_patch.py --dir path/to/smolvla # patches a given folder
    python fox_patch.py --revert              # restores the .orig backups
"""
import argparse
import importlib.util
import pathlib
import shutil
import sys


def find_dir() -> pathlib.Path:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("lerobot is not installed. pip install 'lerobot[smolvla]' first.")
    return pathlib.Path(list(spec.submodule_search_locations)[0]) / "policies" / "smolvla"


def sub(text: str, old: str, new: str, name: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"[patch failed] anchor '{name}' found {n}x (expected 1). "
                 f"Your lerobot version differs from the one this patch was written for.")
    return text.replace(old, new)


# --------------------------------------------------------------------------------------
# configuration_smolvla.py
# --------------------------------------------------------------------------------------
def patch_config(t: str) -> str:
    return sub(
        t,
        '    compile_mode: str = "max-autotune"  # Torch compile mode\n',
        '    compile_mode: str = "max-autotune"  # Torch compile mode\n'
        "\n"
        "    # FoX (Forgetting Attention) in the action expert's self-attention layers\n"
        "    use_fox: bool = False\n"
        "    fox_bias_init: float = 4.0  # sigmoid(4) = 0.982\n"
        "    fox_identity_init: bool = True  # alpha = 0 -> exactly the baseline at step 0\n"
        "    fox_alpha_lr: float = 2e-3  # Adam moves ~LR per step, so alpha needs a larger LR than 1e-4\n",
        "config.fields",
    )


# --------------------------------------------------------------------------------------
# smolvlm_with_expert.py
# --------------------------------------------------------------------------------------
FORGET_GATE = '''import torch.nn.functional as F  # noqa: E402


class ForgetGate(nn.Module):
    """FoX forget gate. Returns an additive attention bias alpha_h * D, D_ij = c_i - c_j,
    c = cumsum(log f), log f = logsigmoid(W h + b), computed in fp32, one gate per query head.
    alpha_h is clamped >= 0 and initialised to 0 (identity at init) when identity_init=True."""

    def __init__(self, hidden, heads, bias_init=4.0, identity_init=True):
        super().__init__()
        self.proj = nn.Linear(hidden, heads)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, bias_init)
        self.alpha = nn.Parameter(torch.zeros(heads) if identity_init else torch.ones(heads))

    def forward(self, x):  # x: [B, Ls, E] normed expert hidden states
        log_f = F.logsigmoid(self.proj(x.float()))  # [B, Ls, H]
        c = torch.cumsum(log_f, dim=1).transpose(1, 2)  # [B, H, Ls]
        d = c[:, :, :, None] - c[:, :, None, :]  # [B, H, Ls, Ls], zero diagonal
        return self.alpha.clamp(min=0)[None, :, None, None] * d


class SmolVLMWithExpertModel(nn.Module):'''

GATE_BUILD = '''        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        # FoX gates: one per self-attention layer of the action expert
        self.fox_gates = nn.ModuleDict()
        if use_fox:
            assert self.num_expert_layers == self.num_vlm_layers, (
                "FoX gates assume num_expert_layers == num_vlm_layers (default config)"
            )
            for _i in range(self.num_vlm_layers):
                _is_self = ("cross" not in attention_mode) or (
                    self_attn_every_n_layers > 0 and _i % self_attn_every_n_layers == 0
                )
                if _is_self:
                    self.fox_gates[str(_i)] = ForgetGate(
                        lm_expert_config.hidden_size,
                        config.text_config.num_attention_heads,
                        fox_bias_init,
                        fox_identity_init,
                    )
'''


def patch_vlm(t: str) -> str:
    t = sub(t, "class SmolVLMWithExpertModel(nn.Module):", FORGET_GATE, "class")
    t = sub(t, 'torch_dtype="bfloat16",', 'torch_dtype="float32",', "fp32 loader")
    t = sub(
        t,
        '        device: str = "auto",\n    ):\n        super().__init__()\n        require_package("transformers", extra="smolvla")',
        '        device: str = "auto",\n'
        "        use_fox: bool = False,\n"
        "        fox_bias_init: float = 4.0,\n"
        "        fox_identity_init: bool = True,\n"
        '    ):\n        super().__init__()\n        require_package("transformers", extra="smolvla")',
        "init signature",
    )
    t = sub(
        t,
        "        # Remove unused embed_tokens\n        self.lm_expert.embed_tokens = None\n",
        GATE_BUILD,
        "gate build",
    )
    # forward_attn_layer: init gate_bias
    t = sub(
        t,
        "        query_states = []\n        key_states = []\n        value_states = []\n",
        "        gate_bias = None\n        query_states = []\n        key_states = []\n        value_states = []\n",
        "gate_bias init",
    )
    # forward_attn_layer: compute the gate from the normed expert hidden state
    t = sub(
        t,
        "            hidden_states = layer.input_layernorm(hidden_states)\n",
        "            hidden_states = layer.input_layernorm(hidden_states)\n"
        "            if i == 1 and str(layer_idx) in self.fox_gates:  # expert (action) tokens only\n"
        "                gate_bias = self.fox_gates[str(layer_idx)](hidden_states)\n",
        "gate compute",
    )
    # forward_attn_layer: pad the [B,H,Ls,Ls] bias to [B,H,Lq,Lk] (zeros on prefix rows/cols) and pass it on
    t = sub(
        t,
        "        att_output = attention_interface(\n"
        "            attention_mask_, batch_size, head_dim, query_states, key_states, value_states\n"
        "        )\n",
        "        att_bias = None\n"
        "        if gate_bias is not None:\n"
        "            _ls, _lq, _lk = gate_bias.shape[-1], query_states.shape[1], key_states.shape[1]\n"
        "            att_bias = F.pad(gate_bias, (_lk - _ls, 0, _lq - _ls, 0))\n"
        "        att_output = attention_interface(\n"
        "            attention_mask_, batch_size, head_dim, query_states, key_states, value_states, att_bias=att_bias\n"
        "        )\n",
        "attention call",
    )
    # eager attention: accept and add the bias
    t = sub(
        t,
        "    def eager_attention_forward(\n"
        "        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states\n"
        "    ):\n",
        "    def eager_attention_forward(\n"
        "        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states, att_bias=None\n"
        "    ):\n",
        "eager signature",
    )
    t = sub(
        t,
        "        att_weights = att_weights.to(dtype=torch.float32)\n        big_neg",
        "        att_weights = att_weights.to(dtype=torch.float32)\n"
        "        if att_bias is not None:\n"
        "            att_weights = att_weights + att_bias\n"
        "        big_neg",
        "eager bias add",
    )
    return t


# --------------------------------------------------------------------------------------
# modeling_smolvla.py
# --------------------------------------------------------------------------------------
OPTIM = '''    def get_optim_params(self):
        if not getattr(self.config, "use_fox", False):
            return self.parameters()
        alpha, gate_w, rest = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "fox_gates" in n:
                (alpha if n.endswith(".alpha") else gate_w).append(p)
            else:
                rest.append(p)
        groups = [{"params": rest}]
        if gate_w:  # base LR, no weight decay
            groups.append({"params": gate_w, "weight_decay": 0.0})
        if alpha:  # larger LR, no weight decay
            groups.append({"params": alpha, "lr": self.config.fox_alpha_lr, "weight_decay": 0.0})
        return groups'''


def patch_modeling(t: str) -> str:
    t = sub(
        t,
        "    def get_optim_params(self) -> dict:\n        return self.parameters()",
        OPTIM,
        "optim params",
    )
    t = sub(
        t,
        '            device=self.config.device if self.config.device is not None else "auto",\n        )\n',
        '            device=self.config.device if self.config.device is not None else "auto",\n'
        "            use_fox=self.config.use_fox,\n"
        "            fox_bias_init=self.config.fox_bias_init,\n"
        "            fox_identity_init=self.config.fox_identity_init,\n"
        "        )\n",
        "pass flags",
    )
    t = sub(
        t,
        "        self.prefix_length = self.config.prefix_length\n        self.rtc_processor = rtc_processor\n",
        "        self.prefix_length = self.config.prefix_length\n        self.rtc_processor = rtc_processor\n"
        "        self.float()  # fp32 everywhere (T4 has no native bf16)\n",
        "fp32 cast",
    )
    return t


PATCHES = {
    "configuration_smolvla.py": patch_config,
    "smolvlm_with_expert.py": patch_vlm,
    "modeling_smolvla.py": patch_modeling,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    d = pathlib.Path(args.dir) if args.dir else find_dir()
    print("SmolVLA dir:", d)

    if args.revert:
        for name in PATCHES:
            bak = d / (name + ".orig")
            if bak.exists():
                shutil.copy(bak, d / name)
                print("restored", name)
        return

    if "use_fox" in (d / "configuration_smolvla.py").read_text():
        print("Already patched, nothing to do.")
        return

    new = {}
    for name, fn in PATCHES.items():  # patch in memory first, write only if ALL succeed
        new[name] = fn((d / name).read_text())
    for name, text in new.items():
        shutil.copy(d / name, d / (name + ".orig"))
        (d / name).write_text(text)
        print("patched", name)


if __name__ == "__main__":
    main()
