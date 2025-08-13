######################################################################
#
# Copyright (c) 2025 Shopee Inc. All Rights Reserved.
#
######################################################################

# """
# file: test_group_gemm.py
# author: min.yang@shopee.com, yangfan.bai@shopee.com, finch.li@shopee.com
# date: 2025-08-08 16:20:00
# brief: group gemm test file.
# """

import torch
import time
from typing import List, Tuple
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
import argparse
import json
import os


class TEGPUGroupGemmTester:
    def __init__(
        self, group_config, dtype=torch.float16, accumulate=False, transa=False, transb=False
    ):
        self.dtype = dtype
        self.accumulate = accumulate

        self.num_groups = len(group_config)

        self.group_config = group_config

        self.m_splits = []
        for i in range(self.num_groups):
            self.m_splits.append(group_config[i][0])

        self.transa = transa
        self.transb = transb
        self.device = torch.cuda.get_device_name(0)

    def generate_inputs_outputs(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        A_list, B_list, C_list, ref_C_list = [], [], [], []
        for n, m, k in self.group_config:
            A = torch.randn(
                k if not self.transa else m,
                m if not self.transa else k,
                dtype=self.dtype,
                device="cuda",
            )
            B = torch.randn(
                n if not self.transb else k,
                k if not self.transb else n,
                dtype=self.dtype,
                device="cuda",
            )
            ref_C = torch.randn(n, m, dtype=self.dtype, device="cuda")
            C = ref_C.clone()
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
            ref_C_list.append(ref_C)
        return A_list, B_list, C_list, ref_C_list

    def test_grouped_gemm(
        self, atol=1e-2, rtol=1e-2, check_accuracy=True, check_performance=False
    ):

        WARM_ITERS = 10
        ITERS = 1000

        A, B, C, ref_C = self.generate_inputs_outputs()

        if not self.transa and not self.transb:
            layout = "NN"
        elif not self.transa:
            layout = "NT"
        elif not self.transb:
            layout = "TN"
        else:
            print("Not Support TT")
            return

        torch.cuda.synchronize()
        general_grouped_gemm(
            A,
            B,
            C,
            self.dtype,
            get_multi_stream_cublas_workspace(),
            layout=layout,
            m_splits=self.m_splits,
            accumulate=self.accumulate
        )
        torch.cuda.synchronize()

        print(f'\n=== Accuracy Testing with Layout:{layout} GemmType:{os.getenv("NVTE_USE_CUTLASS_GROUPGEMM", "0")}')
        if check_accuracy:

            alpha = 1.0
            beta = 1.0 if self.accumulate else 0.0

            ref_out = [
                alpha * torch.matmul(b if not self.transb else b.T, a if not self.transa else a.T)
                + beta * c
                for b, a, c in zip(B, A, ref_C)
            ]

            max_abs_err = []
            max_rel_err = []
            for i in range(self.num_groups):
                abs_err = (ref_out[i] - C[i]).abs()
                print(ref_out[i].shape)
                print(C[i].shape)
                rel_err = abs_err / (ref_out[i].abs() + 1e-5)
                max_abs_err.append(abs_err.max().item())
                max_rel_err.append(rel_err.max().item())

                if not torch.allclose(C[i], ref_out[i], atol=atol, rtol=rtol):
                    print(f"[Group {i}] ❌ Mismatch:")
                    print(f"  Max Abs Error = {max_abs_err[-1]:.6f}")
                    print(f"  Max Rel Error = {max_rel_err[-1]:.6f}")
                else:
                    print(
                        f"[Group {i}] ✅ Passed: max abs error = {max_abs_err[-1]:.6f}, max rel"
                        f" error = {max_rel_err[-1]:.6f}"
                    )

        if check_performance:
            print("\n=== Performance Testing ===")
            # warm up
            torch.cuda.synchronize()
            for _ in range(WARM_ITERS):
                general_grouped_gemm(
                    A,
                    B,
                    C,
                    self.dtype,
                    get_multi_stream_cublas_workspace(),
                    layout=layout,
                    m_splits=self.m_splits,
                    accumulate=self.accumulate
                )

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(ITERS):
                general_grouped_gemm(
                    A,
                    B,
                    C,
                    self.dtype,
                    get_multi_stream_cublas_workspace(),
                    layout=layout,
                    m_splits=self.m_splits,
                    accumulate=self.accumulate
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            exec_time = end_time - start_time

            avg_time_ms = exec_time * 1000 / ITERS

            # FLOPs 计算
            total_flops = sum([2 * m * n * k for (m, n, k) in self.group_config])
            tflops = total_flops / (avg_time_ms * 1e9)

            print(f"🔥 avgerage used time: {avg_time_ms:.3f} ms/iter")
            print(f"⚡ average throughput: {tflops:.2f} TFLOPs")


def run_grouped_gemm(group_config, check_performance, transa, transb, accumulate):
    print(f"🔧 Running grouped GEMM with:")
    print(f"  group_config = {group_config}")
    print(f"  check_performance = {check_performance}")
    print(f"  transa = {transa}, transb = {transb}")
    print("-" * 50)
    tester = TEGPUGroupGemmTester(group_config, transa=transa, transb=transb, accumulate=accumulate)
    tester.test_grouped_gemm(check_performance=check_performance)


if __name__ == "__main__":

    config_data = {
        "configs": [
            {
                "group_config": [
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048],
                    [4096, 768, 2048]
                ],
                "accumulate": False,
                "check_performance": True,
                "transa": False,
                "transb": True
            },
            {
                "group_config": [
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048],
                    [2048, 768, 2048]
                ],
                "accumulate": False,
                "check_performance": True,
                "transa": False,
                "transb": True
            }
        ]
    }
    
    for i, case in enumerate(config_data["configs"]):
        group_config = [tuple(x) for x in case["group_config"]]
        accumulate = case.get("accumulate", False)
        check_performance = case.get("check_performance", False)
        # NOTE(Alan): for cublas weight is A, input is B
        #             so we need to swap A B
        transb = case.get("transa", False)
        transa = case.get("transb", False)

        print(f"\n🧪 Case {i+1}:")
        run_grouped_gemm(group_config, check_performance, transa, transb, accumulate)
