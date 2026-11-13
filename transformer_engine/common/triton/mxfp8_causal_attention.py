# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for causal MXFP8 attention without future-token scale leakage.

A columnwise MXFP8 V scale is shared by 32 consecutive tokens, so under an ordinary causal
mask a query can read a V value whose scale was raised by a later token in its own 32-token
block (https://github.com/NVIDIA/TransformerEngine/issues/3248). The forward therefore
splits each query's keys in two:

* keys in completed 32-token blocks, ``j < 32 * floor(i / 32)``: cuDNN's MXFP8 SDPA with
  ``causal_exclude_own_block=32``, whose V scales cover visible tokens only;
* keys in the query's own block: :func:`own_block_attention_kernel`, which multiplies the
  softmax weights by a token-normalized copy of V (one power-of-two amplitude per token,
  from :func:`token_values_kernel`) and merges the result with the first partial through
  the two log-sum-exps.

:func:`decode_kernel` prepares the backward: BF16 Q and K decoded from their rowwise MXFP8
bytes and V decoded from the token-normalized copy, so no backward operand depends on a
future token either.

All kernels take D = 128, E4M3 payloads, and rowwise scales in the F8_128x4 layout of each
(batch, head) matrix: a 128-token tile holds 512 E8M0 bytes, token ``t`` and 32-feature
group ``g`` at ``(t % 32) * 16 + (t % 128 // 32) * 4 + g``.
"""

import triton
import triton.language as tl

ZERO_TOKEN = tl.constexpr(-1000)  # amplitude exponent of an all-zero V row
LOG2_E = tl.constexpr(1.4426950408889634)
LOG2_E4M3_MAX = tl.constexpr(8.807354922057604)  # log2(448)


@triton.jit
def _scale_offset(token, group, heads_index, num_tokens: tl.constexpr):
    """Byte offset of a rowwise E8M0 scale in the F8_128x4 layout."""
    tile = heads_index * (num_tokens // 128) + token // 128
    return tile * 512 + (token % 32) * 16 + (token % 128 // 32) * 4 + group


@triton.jit
def _decode_scale(code):
    """E8M0 code to its power-of-two value; code 0 is 2**-127 (an FP32 subnormal)."""
    code = code.to(tl.int32)
    return tl.where(code == 0, 0x00400000, code << 23).to(tl.float32, bitcast=True)


@triton.jit
def token_values_kernel(
    V,
    U,
    E,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    num_tokens: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    """Token-normalized V: ``U = V * 2**(8 - E)`` in E4M3, one exponent ``E`` per token.

    ``E = ceil(log2(max |V_t|))`` (bounded below by -126), so ``|U| <= 256`` and the cast to
    E4M3 depends on the token's own values only. ``U`` is ``[B, H_kv, S, 128]`` and ``E`` is
    ``[B, H_kv, S]``; an all-zero row stores ``U = 0`` and ``E = ZERO_TOKEN``.
    """
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = block * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    v = tl.load(
        V + batch * v_stride_b + t[:, None] * v_stride_s + head * v_stride_h + d[None, :]
    ).to(tl.float32)
    maximum = tl.max(tl.abs(v), 1)
    bits = maximum.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    # Exact v * 2**(8 - e) as two normal power-of-two factors (8 - e lies in [-120, 135]).
    shift = 8 - e
    first = (((shift >> 1) + 127) << 23).to(tl.float32, bitcast=True)
    second = ((shift - (shift >> 1) + 127) << 23).to(tl.float32, bitcast=True)
    u = (v * first[:, None] * second[:, None]).to(tl.float8e4nv)
    u = tl.where(maximum[:, None] == 0, 0, u)
    row = (batch * num_kv_heads + head) * num_tokens + t
    tl.store(U + row[:, None] * 128 + d[None, :], u)
    tl.store(E + row, tl.where(maximum == 0, ZERO_TOKEN, e))


@triton.jit
def own_block_attention_kernel(
    Q,
    K,
    SQ,
    SK,
    U,
    E,
    O,
    LSE,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    o_stride_b,
    o_stride_s,
    o_stride_h,
    attn_scale,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    PACK: tl.constexpr,
):
    """Attention of each query over its own 32-token block, merged into the completed-block
    partial.

    ``O`` and ``LSE`` hold the completed-block output and natural-log softmax statistics on
    entry (rows 0..31: O = 0, LSE = -inf) and the full causal result on exit; each program
    reads its rows before overwriting them. One program covers 128 MMA rows: 128 / PACK
    tokens for each of PACK query heads that share a KV head.
    """
    tile, packed_head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    KEYS: tl.constexpr = 128 // PACK
    GROUPS: tl.constexpr = KEYS // 32
    kv_head = packed_head // ((num_heads // num_kv_heads) // PACK)
    row = tl.arange(0, 128)
    head = packed_head * PACK + row // KEYS
    token = tile * KEYS + row % KEYS
    key = tile * KEYS + tl.arange(0, KEYS)
    d = tl.arange(0, 128)
    g = tl.arange(0, 4)

    # Issue the partial loads first; they overlap the two matrix products.
    o_offset = batch * o_stride_b + token[:, None] * o_stride_s + head[:, None] * o_stride_h
    partial = tl.load(O + o_offset + d[None, :]).to(tl.float32)
    lse_offset = (batch * num_heads + head) * num_tokens + token
    partial_lse = tl.load(LSE + lse_offset)
    q = tl.load(
        Q + batch * q_stride_b + token[:, None] * q_stride_s + head[:, None] * q_stride_h + d
    )
    k = tl.load(
        K + batch * k_stride_b + key[:, None] * k_stride_s + kv_head * k_stride_h + d[None, :]
    )
    sq = tl.load(
        SQ
        + _scale_offset(token[:, None], g[None, :], batch * num_heads + head[:, None], num_tokens)
    )
    sk = tl.load(
        SK + _scale_offset(key[:, None], g[None, :], batch * num_kv_heads + kv_head, num_tokens)
    )
    score = tl.dot_scaled(q, sq, "e4m3", tl.trans(k), sk, "e4m3") * attn_scale

    # Each row sees only its own 32-key group: select it so the softmax runs on 128 x 32.
    j = tl.arange(0, 32)
    pick = tl.arange(0, GROUPS)[None, :, None] == ((row % KEYS) // 32)[:, None, None]
    if GROUPS == 1:
        own_score = score
        e = tl.load(E + (batch * num_kv_heads + kv_head) * num_tokens + key)[None, :]
    else:
        own_score = tl.sum(tl.where(pick, tl.reshape(score, (128, GROUPS, 32)), 0.0), 1)
        e = tl.load(
            E
            + (batch * num_kv_heads + kv_head) * num_tokens
            + (token[:, None] // 32) * 32
            + j[None, :]
        )
    visible = j[None, :] <= (token % 32)[:, None]
    own_score = tl.where(visible, own_score, -float("inf"))
    m = tl.max(own_score, 1)
    p = tl.exp(own_score - m[:, None])
    denominator = tl.sum(p, 1)

    # W = P * 2**E in E4M3 with one E8M0 scale per row: P V = (W * 2**-8) U exactly in the
    # scales, with the only rounding in W and U themselves.
    log_w = (own_score - m[:, None]) * LOG2_E + e
    log_w = tl.where(visible & (e != ZERO_TOKEN), log_w, -float("inf"))
    w_max = tl.max(log_w, 1)
    w_exponent = tl.where(w_max == -float("inf"), 0, tl.ceil(w_max - LOG2_E4M3_MAX))
    w_exponent = tl.minimum(127, tl.maximum(-127, w_exponent))
    own_w = tl.exp2(log_w - w_exponent[:, None])
    if GROUPS == 1:
        w = own_w.to(tl.float8e4nv)
    else:
        # Block-diagonal W: zero outside the row's own key group.
        w = tl.reshape(tl.where(pick, own_w[:, None, :], 0.0), (128, KEYS)).to(tl.float8e4nv)
    w_scale = tl.broadcast_to((w_exponent + 127).to(tl.uint8)[:, None], (128, GROUPS))
    u = tl.load(U + ((batch * num_kv_heads + kv_head) * num_tokens + key[:, None]) * 128 + d)
    u_scale = tl.full((128, GROUPS), 127 - 8, tl.uint8)  # 2**-8 for every 32 keys of U
    numerator = tl.dot_scaled(w, w_scale, "e4m3", u, u_scale, "e4m3")

    # The partial is normalized, so exp(partial_lse) is its denominator. Rows 0..31 have no
    # completed block; discard their partial before any arithmetic.
    partial_lse = tl.where(token >= 32, partial_lse, -float("inf"))
    partial = tl.where(token[:, None] >= 32, partial, 0.0)
    common = tl.maximum(m, partial_lse)
    a, z = tl.exp(partial_lse - common), tl.exp(m - common)
    total = a + z * denominator
    result = (a[:, None] * partial + z[:, None] * numerator) / total[:, None]
    tl.store(O + o_offset + d[None, :], result.to(O.dtype.element_ty))
    tl.store(LSE + lse_offset, common + tl.log(total))


@triton.jit
def decode_kernel(
    Q8,
    SQ,
    K8,
    SK,
    U,
    E,
    Q,
    K,
    V,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    num_batches: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward operands: Q and K decoded from rowwise MXFP8, V from the token-normalized copy.

    Programs ``[0, B * H)`` along axis 1 decode Q; the next ``B * H_kv`` decode K and V. The
    decoded tensors use the strides of the MXFP8 payloads (Q like Q8; K and V like K8). Every
    value is exact in BF16 or FP16 barring underflow: an E4M3 value times a power of two.
    """
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    matrix = tl.program_id(1)
    d = tl.arange(0, 128)
    g = d // 32
    if matrix < num_batches * num_heads:
        batch, head = matrix // num_heads, matrix % num_heads
        offset = batch * q_stride_b + t[:, None] * q_stride_s + head * q_stride_h + d[None, :]
        code = tl.load(SQ + _scale_offset(t[:, None], g[None, :], matrix, num_tokens))
        q = tl.load(Q8 + offset).to(tl.float32) * _decode_scale(code)
        tl.store(Q + offset, q.to(Q.dtype.element_ty))
    else:
        matrix -= num_batches * num_heads
        batch, head = matrix // num_kv_heads, matrix % num_kv_heads
        offset = batch * k_stride_b + t[:, None] * k_stride_s + head * k_stride_h + d[None, :]
        code = tl.load(SK + _scale_offset(t[:, None], g[None, :], matrix, num_tokens))
        k = tl.load(K8 + offset).to(tl.float32) * _decode_scale(code)
        tl.store(K + offset, k.to(K.dtype.element_ty))
        row = matrix * num_tokens + t
        amplitude = tl.exp2((tl.load(E + row) - 8).to(tl.float32))
        v = tl.load(U + row[:, None] * 128 + d[None, :]).to(tl.float32) * amplitude[:, None]
        tl.store(V + offset, v.to(V.dtype.element_ty))
