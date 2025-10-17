/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "./config.h"

#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstring>

#include "../util/logging.h"

NVTEMatmulConfig nvte_create_matmul_config() { return new transformer_engine::MatmulConfig; }

void nvte_get_matmul_config_attribute(NVTEMatmulConfig config, NVTEMatmulConfigAttribute attr,
                                      void *buf, size_t size_in_bytes, size_t *size_written) {
  // Write attribute size
  NVTE_CHECK(attr < kNVTEMatmulConfigNumAttributes, "Invalid NVTEMatmulConfigAttribute (got ",
             static_cast<int>(attr), ")");
  NVTE_CHECK(size_written != nullptr, "Invalid size_written (got NULL)");
  const auto &attr_size = transformer_engine::MatmulConfig::attr_sizes[attr];
  *size_written = attr_size;

  // Return immediately if buffer is not provided
  if (buf == nullptr) {
    return;
  }

  // Check buffer size
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for matmul config attribute "
             "(attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");

  // Write to buffer
  NVTE_CHECK(config != nullptr, "Invalid NVTEMatmulConfig (got NULL)");
  const auto &config_ = *reinterpret_cast<const transformer_engine::MatmulConfig *>(config);
  switch (attr) {
    case kNVTEMatmulConfigBiasTensor:
      std::memcpy(buf, &config_.bias_tensor, attr_size);
      break;
    case kNVTEMatmulConfigDBiasTensor:
      std::memcpy(buf, &config_.dbias_tensor, attr_size);
      break;
    case kNVTEMatmulConfigWithGELUEpilogue:
      std::memcpy(buf, &config_.with_gelu_epilogue, attr_size);
      break;
    case kNVTEMatmulConfigWithDGELUEpilogue:
      std::memcpy(buf, &config_.with_dgelu_epilogue, attr_size);
      break;
    case kNVTEMatmulConfigEpilogueAuxTensor:
      std::memcpy(buf, &config_.epilogue_aux_tensor, attr_size);
      break;
    case kNVTEMatmulConfigUseSplitAccumulator:
      std::memcpy(buf, &config_.use_split_accumulator, attr_size);
      break;
    case kNVTEMatmulConfigSMCount:
      std::memcpy(buf, &config_.sm_count, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEMatmulConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_matmul_config_attribute(NVTEMatmulConfig config, NVTEMatmulConfigAttribute attr,
                                      const void *buf, size_t size_in_bytes) {
  // Check attribute and buffer
  NVTE_CHECK(attr < kNVTEMatmulConfigNumAttributes, "Invalid NVTEMatmulConfigAttribute (got ",
             static_cast<int>(attr), ")");
  const auto &attr_size = transformer_engine::MatmulConfig::attr_sizes[attr];
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for matmul config attribute "
             "(attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");
  NVTE_CHECK(buf != nullptr, "Invalid buffer (got NULL)");

  // Read from buffer
  NVTE_CHECK(config != nullptr, "Invalid NVTEMatmulConfig (got NULL)");
  auto &config_ = *reinterpret_cast<transformer_engine::MatmulConfig *>(config);
  switch (attr) {
    case kNVTEMatmulConfigBiasTensor:
      std::memcpy(&config_.bias_tensor, buf, attr_size);
      break;
    case kNVTEMatmulConfigDBiasTensor:
      std::memcpy(&config_.dbias_tensor, buf, attr_size);
      break;
    case kNVTEMatmulConfigWithGELUEpilogue:
      std::memcpy(&config_.with_gelu_epilogue, buf, attr_size);
      break;
    case kNVTEMatmulConfigWithDGELUEpilogue:
      std::memcpy(&config_.with_dgelu_epilogue, buf, attr_size);
      break;
    case kNVTEMatmulConfigEpilogueAuxTensor:
      std::memcpy(&config_.epilogue_aux_tensor, buf, attr_size);
      break;
    case kNVTEMatmulConfigUseSplitAccumulator:
      std::memcpy(&config_.use_split_accumulator, buf, attr_size);
      break;
    case kNVTEMatmulConfigSMCount:
      std::memcpy(&config_.sm_count, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEMatmulConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_destroy_matmul_config(NVTEMatmulConfig config) {
  if (config != nullptr) {
    delete reinterpret_cast<transformer_engine::MatmulConfig *>(config);
  }
}
