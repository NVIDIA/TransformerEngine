from __future__ import annotations
import torch
from .persistent import Persistent
from .recipe import Recipe

FP8Meta = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PersistentFP8Meta(Persistent[FP8Meta]):
    amaxes: torch.Tensor  # (amax_history_len, num_tensors)
    scaling_factors: torch.Tensor  # (num_tensors,)
    scaling_factors_inversed: torch.Tensor  # (num_tensors,)

    def _generate(self):
        if self._iteration() == 1:
            if self._is_new_iteration():
                # Allocate first iteration metatensors
                self._one = torch.ones(1, device="cuda")
                self._first_iteration_amaxes: list[torch.Tensor] = []
            amax = torch.zeros(1, device="cuda")
            self._first_iteration_amaxes.append(amax)
            self._index_within_iteration()  # increment tensor index
            return (amax, self._one, self._one)
        else:
            if self._iteration() == 2 and self._is_new_iteration():
                # Allocate metatensors
                self.amaxes = torch.zeros(
                    (Recipe.current().amax_history_len, self._max_index()),
                    device="cuda",
                )
                self.scaling_factors = torch.ones(self._max_index(), device="cuda")
                self.scaling_factors_inversed = torch.ones(
                    self._max_index(), device="cuda"
                )
                # Copy amaxes from first iteration
                self.amaxes[0] = torch.cat(self._first_iteration_amaxes)
                # Delete first iteration amaxes
                del self._first_iteration_amaxes
            if self._iteration() % Recipe.current().amax_reduction_period == 0:
                amaxes_t = self.amaxes.T  # (num_tensors, amax_history_len)
                reduced = Recipe.current().amax_reduction_method(
                    amaxes_t
                )  # (num_tensors,)
                Recipe.current().scaling_factor_compute_method(
                    reduced, self.scaling_factors
                )
                torch.reciprocal(
                    self.scaling_factors, out=self.scaling_factors_inversed
                )
            tensor_idx = self._index_within_iteration()
            return (
                self.amaxes[
                    self._iteration() % Recipe.current().amax_history_len, tensor_idx
                ],
                self.scaling_factors[tensor_idx],
                self.scaling_factors_inversed[tensor_idx],
            )
