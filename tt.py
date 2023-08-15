import torch
from typing import Any, Callable, Optional, Tuple, Union, List
from itertools import repeat

dtype = torch.float16

def get_qkv(qkv_layout):
    dim_to_num = {'b':4, 's':128, 'h':16, 'd':512, '3':3, '2':2, 't':4*128}
    inp = []
    for i,layout in enumerate(qkv_layout.split('_')):
        tensor_shape = [dim_to_num[j] for j in layout]
        tensor = 0.1 * torch.randn(tensor_shape, dtype = dtype).cuda()
        tensor_count = 1
        split_dim = 0
        for dim,l in enumerate(layout):
             if l.isdigit():
                 tensor_count = int(l)
                 split_dim = dim
                 break
        tensors = torch.split(tensor, 1, dim = split_dim) if split_dim != 0 else [tensor]
        for j in range(tensor_count):
            inp.append(tensors[j])
    for i in range(3):
        inp[i].requires_grad=True
    return inp[0], inp[1], inp[2]


def _get_qkv_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_format: str = 'sbhd',
    ) -> Tuple[Union[str, List, Tuple, None], ...]:

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"

    data_ptr = q.untyped_storage().data_ptr()
    check_ptrs_qkv = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
    data_ptr = k.untyped_storage().data_ptr()
    check_ptrs_kv = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])

    stride = q.stride()
    check_strides_qkv = all(stride == x.stride() for x in [q, k, v])
    stride = k.stride()
    check_strides_kv = all(stride == x.stride() for x in [k, v])

    shape = q.shape
    check_shapes_qkv = all(shape == x.shape for x in [q, k, v])
    shape = k.shape
    check_shapes_kv = all(shape == x.shape for x in [k, v])

    last_dim_size = q.shape[-1]
    check_last_dim_offsets_qkv = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    last_dim_size = k.shape[-1]
    check_last_dim_offsets_kv = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([k, v]))

    last_two_dims_size = q.shape[-1] * q.shape[-2]
    check_last_two_dims_offsets_qkv = all(i * last_two_dims_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    last_two_dims_size = k.shape[-1] * k.shape[-2]
    check_last_two_dims_offsets_kv = all(i * last_two_dims_size == x.storage_offset()
                        for i, x in enumerate([k, v]))

    qkv_layout = None
    if (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
        and check_last_two_dims_offsets_qkv
        and not check_last_dim_offsets_qkv):
        # sb3hd, bs3hd, t3hd
        qkv_layout = qkv_format[:-2] + '3' + qkv_format[-2:]
    elif check_ptrs_qkv and check_strides_qkv and check_shapes_qkv and check_last_dim_offsets_qkv:
        # sbh3d, bsh3d, th3d
        qkv_layout = qkv_format[:-1] + '3' + qkv_format[-1:]
    elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
        and check_last_two_dims_offsets_kv
        and not check_last_dim_offsets_kv):
        # sbhd_sb2hd, bshd_bs2hd, thd_t2hd
        qkv_layout = qkv_format + '_' + qkv_format[:-2] + '2' + qkv_format[-2:]
    elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
        and check_last_dim_offsets_kv):
        # sbhd_sbh2d, bshd_bsh2d, thd_th2d
        qkv_layout = qkv_format + '_' + qkv_format[:-1] + '2' + qkv_format[-1:]
    elif check_strides_kv and check_shapes_kv:
        # sbhd_sbhd_sbhd, bshd_bshd_bshd, thd_thd_thd
        qkv_layout = '_'.join(list(repeat(qkv_format, 3)))
    else:
        raise Exception("The provided qkv memory layout is not supported!")
    
    return qkv_layout


qkv_layouts = [
    'sb3hd', 'sbh3d', 'sbhd_sb2hd', 'sbhd_sbh2d', 'sbhd_sbhd_sbhd',
    'bs3hd', 'bsh3d', 'bshd_bs2hd', 'bshd_bsh2d', 'bshd_bshd_bshd',
    't3hd', 'th3d', 'thd_t2hd', 'thd_th2d', 'thd_thd_thd',
    ]

for qkv_layout in qkv_layouts:
    qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
    q, k, v = get_qkv(qkv_layout)
    qkv_lay = _get_qkv_layout(q, k, v, qkv_format)
    print(qkv_layout, qkv_lay, qkv_layout == qkv_lay)

