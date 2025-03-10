# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

class InferenceParams:  # pylint: disable=too-few-public-methods
    """
    Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context during inference.

    Parameters
    ----------
    max_batch_size : int
                    maximum batch size during inference.
    max_sequence_length : int
                         maximum sequence length during inference.
    """

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_indices):
        """
        Reorders the KV cache using the specified batch indices.

        Parameters
        ----------
        batch_indices : List[int]
                       Sequence of indices to reorder along the batch dimensions of
                       the KV cache. Must have a length equal to the batch size.
        """
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number, inference_memory in self.key_value_memory_dict.items():
            inference_key_memory, inference_value_memory = inference_memory
            assert (
                len(batch_indices) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_indices]
            new_inference_value_memory = inference_value_memory[:, batch_indices]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )