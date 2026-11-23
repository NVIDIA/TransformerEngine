# Transformer Engine vX.YZ Release Notes

## Key Features and Enhancements

- 

## Fixed Issues

- 

## Breaking Changes in This Release

- For TE 3.0, remove the legacy split-accumulator option from GEMM APIs and
  enable accurate FP8 accumulation automatically on Hopper. Remove
  `use_split_accumulator` arguments, `transformer_engine.common.recipe.MMParams`,
  and the recipe fields `fp8_gemm_fprop`, `fp8_gemm_dgrad`, and `fp8_gemm_wgrad`
  from callers. Native consumers must rebuild against the updated C/C++ APIs.
  The JAX environment variable `NVTE_FP8_GEMM_HIGH_PRECISION_ACCUMULATION` is
  also removed; accurate accumulation is always used.

## Deprecated Features

There are no deprecated features in this release.
