"""Attention-only Baguette KV cache with correct masks for verification blocks.

Each attention layer stores K and V as [batch, n_kv_head, max_len, head_dim].
The logical cache position is an argument: after a rejected proposal, writing
the correction at the rejection position overwrites the stale future entries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def allocate(model: torch.nn.Module, max_len: int) -> list[dict]:
    if any(layer.kind != "attn" for layer in model.layers):
        raise ValueError("KV cache currently supports attention-only Baguette models")
    parameter = next(model.parameters())
    return model._alloc_caches(1, max_len, parameter.device, parameter.dtype)


@torch.inference_mode()
def forward_all(model: torch.nn.Module, ids: torch.Tensor, caches: list[dict], pos: int) -> torch.Tensor:
    """Return logits [1, T, vocab] and write T positions into each layer cache."""
    from model import apply_partial_rope, repeat_kv

    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("ids must have shape [1, T] with T >= 1")
    T = ids.shape[1]
    if pos < 0 or pos + T > model.cfg.max_seq_len:
        raise ValueError("cached forward exceeds model context")
    x = model.embed_tokens(ids)
    cos = model.rope_cos[pos:pos + T]
    sin = model.rope_sin[pos:pos + T]

    for layer, cache in zip(model.layers, caches):
        mixer = layer.mixer
        h = layer.input_layernorm(x)
        B = 1
        q = mixer.q_proj(h).view(B, T, mixer.n_head, mixer.head_dim)
        k = mixer.k_proj(h).view(B, T, mixer.n_kv_head, mixer.head_dim)
        v = mixer.v_proj(h).view(B, T, mixer.n_kv_head, mixer.head_dim)
        q, k = mixer.q_norm(q), mixer.k_norm(k)
        q, k = apply_partial_rope(q, k, cos, sin, mixer.rope_dims)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache["k"][:, :, pos:pos + T] = k
        cache["v"][:, :, pos:pos + T] = v
        all_k = cache["k"][:, :, :pos + T]
        all_v = cache["v"][:, :, :pos + T]

        # Query i is at absolute position pos+i; keys after it are invisible.
        query_positions = pos + torch.arange(T, device=ids.device)[:, None]
        key_positions = torch.arange(pos + T, device=ids.device)[None, :]
        causal_mask = key_positions <= query_positions
        attention = F.scaled_dot_product_attention(
            q, repeat_kv(all_k, mixer.n_rep), repeat_kv(all_v, mixer.n_rep),
            attn_mask=causal_mask, dropout_p=0.0,
        )
        attention = attention.transpose(1, 2).reshape(B, T, mixer.n_head * mixer.head_dim)
        if mixer.gate_proj is not None:
            attention = attention * torch.sigmoid(mixer.gate_proj(h))
        x = x + mixer.o_proj(attention)
        x = x + layer.mlp(layer.post_attention_layernorm(x))

    return model.lm_head(model.norm(x))
