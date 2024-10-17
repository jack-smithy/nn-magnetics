from dataclasses import dataclass
from typing import List
from torch import Tensor
import torch


# @dataclass
# class TrainingStats:
#     "shape (epoch, batch, samples)"

#     _train_loss: List[List[Tensor]] = []
#     _test_loss: List[List[Tensor]] = []
#     _angle_accuracy: List[List[Tensor]] = []
#     _amplitude_accuracy: List[List[Tensor]] = []

#     def update(
#         self,
#         train_loss: List[Tensor],
#         test_loss: List[Tensor],
#         angle_accuracy: List[Tensor],
#         amplitude_accuracy: List[Tensor],
#     ) -> None:
#         self._train_loss.append(train_loss)
#         self._test_loss.append(test_loss)
#         self._angle_accuracy.append(angle_accuracy)
#         self._amplitude_accuracy.append(amplitude_accuracy)

#     def get_history(self) -> List[Tensor]:
#         return []
