# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.optimizers import MultiTensorApply

input_size_pairs = [
    (7777 * 77, 555 * 555),
    (777, 555),
    (555, 2048 * 32 + 1),
    (2048 * 32 + 1, 555),
    (555, 2048 * 32),
    (2048 * 32, 555),
    (33333, 555),
    (555, 33333),
]
appliers = [MultiTensorApply(2048 * 32), MultiTensorApply(333), MultiTensorApply(33333)]


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
def test_multi_tensor_scale(input_size_pair, applier, repeat, in_type, out_type, inplace):
    if inplace is True and (out_type is not in_type):
        pytest.skip("inplace=True and out_type != in_type is not supported.")
    elif (in_type == torch.float16 and out_type == torch.bfloat16) or (
        in_type == torch.bfloat16 and out_type == torch.float16
    ):
        pytest.skip("float16 to bfloat16 is not necessary and vice versa.")

    device = torch.device("cuda")
    scale = 4.0
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)
    ref = torch.tensor([1.0], dtype=torch.float32, device=device)
    sizea, sizeb = input_size_pair

    def downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=False):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for i in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

        assert all([torch.allclose(out, ref.to(out_type)) for out in out_list])
        assert overflow_buf.item() == 0

    def find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        t,
        ind,
        val,
        inplace=False,
    ):
        overflow_buf.zero_()
        a = torch.full([sizea], scale, dtype=torch.float32, device=device)
        b = torch.full([sizeb], scale, dtype=torch.float32, device=device)

        out_list = []
        for i in range(repeat):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

        overflow_buf.zero_()
        in_list[t][ind] = val
        applier(tex.multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)
        assert overflow_buf.item() > 0

    downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        0,
        0,
        float("nan"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * repeat - 1,
        sizeb - 1,
        float("inf"),
        inplace=inplace,
    )
    find_inf(
        sizea,
        sizeb,
        applier,
        repeat,
        in_type,
        out_type,
        2 * (repeat // 2),
        sizea // 2,
        float("inf"),
        inplace=inplace,
    )


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_l2norm(input_size_pair, applier, repeat, in_type, per_tensor):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    val = 4.0
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)

    overflow_buf.zero_()
    a = torch.full([sizea], val, dtype=torch.float32, device=device)
    b = torch.full([sizeb], val, dtype=torch.float32, device=device)

    in_list = []
    for i in range(repeat):
        in_list += [a.clone().to(in_type), b.clone().to(in_type)]

    if per_tensor:
        norm, norm_per_tensor = applier(tex.multi_tensor_l2norm, overflow_buf, [in_list], True)
        normab = torch.cat((a.norm().view(1), b.norm().view(1)))
        norm_per_tensor = norm_per_tensor.view(-1, 2)
    else:
        norm, _ = applier(tex.multi_tensor_l2norm, overflow_buf, [in_list], True)

    reference = torch.full(
        [(sizea + sizeb) * repeat], val, dtype=torch.float32, device=device
    ).norm()

    torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
    if per_tensor:
        torch.testing.assert_close(norm_per_tensor, normab.broadcast_to(norm_per_tensor.shape))
    assert overflow_buf.item() == 0


@pytest.mark.parametrize("input_size_pair", input_size_pairs)
@pytest.mark.parametrize("applier", appliers)
@pytest.mark.parametrize("repeat", [1, 55])
@pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_multi_tensor_unscale_l2norm(input_size_pair, applier, repeat, in_type, per_tensor):
    sizea, sizeb = input_size_pair
    device = torch.device("cuda")
    val = 4.0
    inv_scale = 0.5
    inv_scale_cuda = torch.tensor([inv_scale], dtype=torch.float32, device=device)
    overflow_buf = torch.zeros(1, dtype=torch.int32, device=device)

    overflow_buf.zero_()
    a = torch.full([sizea], val, dtype=torch.float32, device=device)
    b = torch.full([sizeb], val, dtype=torch.float32, device=device)

    in_list = []
    for i in range(repeat):
        in_list += [a.clone().to(in_type), b.clone().to(in_type)]

    if per_tensor:
        norm, norm_per_tensor = applier(
            tex.multi_tensor_unscale_l2norm,
            overflow_buf,
            [in_list],
            inv_scale_cuda,
            True,
        )
        normab = torch.cat(((a * inv_scale).norm().view(1), (b * inv_scale).norm().view(1)))
        norm_per_tensor = norm_per_tensor.view(-1, 2)
    else:
        norm, _ = applier(
            tex.multi_tensor_unscale_l2norm,
            overflow_buf,
            [in_list],
            inv_scale_cuda,
            True,
        )

    reference = torch.full(
        [(sizea + sizeb) * repeat], val * inv_scale, dtype=torch.float32, device=device
    ).norm()

    torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
    if per_tensor:
        torch.testing.assert_close(norm_per_tensor, normab.broadcast_to(norm_per_tensor.shape))
    assert overflow_buf.item() == 0
