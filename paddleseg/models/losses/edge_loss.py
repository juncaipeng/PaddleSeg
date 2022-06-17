# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import paddle
import numpy as np
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class EdgeCELoss(nn.Layer):
    """
    Implements the ce loss function for edge pixels.

    Args:
        aperture_size (int, optional): the  aperture size for cv2.Canny.
        kernel_size (int, optional): the kernel size for expand the edge width.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, aperture_size=7, kernel_size=11, ignore_index=255):
        super().__init__()
        self.aperture_size = aperture_size
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.EPS = 1e-8

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        Returns:
            (Tensor): The average loss.
        """
        mask = paddle.cast(label != self.ignore_index, 'float32')
        label_tmp = (label * mask).astype("uint8")
        label_tmp = label_tmp.numpy()[0]
        label_tmp[label_tmp > 0] = 255
        assert label_tmp.ndim == 2, "The ndim of label_tmp should be 2"

        t_lower = 50
        t_upper = 150
        kernel = np.ones((self.kernel_size, self.kernel_size), np.float32)
        edge_mask = cv2.Canny(
            label_tmp, t_lower, t_upper, apertureSize=self.aperture_size)
        edge_mask = cv2.filter2D(edge_mask, -1, kernel)
        edge_mask[edge_mask > 0] = 1
        edge_mask = paddle.to_tensor(edge_mask, dtype="float32")
        mask *= edge_mask
        '''
        mse_loss = F.mse_loss(logit, label, reduction="none")
        avg_loss = paddle.mean(mse_loss) / (paddle.mean(mask) + self.EPS)
        '''

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        loss = F.cross_entropy(
            logit, label, ignore_index=self.ignore_index, reduction='none')
        '''
        loss_tmp = (loss.numpy()[0]).astype("uint8")
        loss_tmp[loss_tmp > 0] = 255
        cv2.imwrite("loss_img_0.jpg", loss_tmp)
        '''

        loss = loss * mask
        '''
        mask = mask.numpy()[0]
        mask[mask > 0] = 255
        cv2.imwrite("mask_img.jpg", mask)

        loss_tmp = (loss.numpy()[0]).astype("uint8")
        loss_tmp[loss_tmp > 0] = 255
        cv2.imwrite("loss_img_1.jpg", loss_tmp)
        '''

        avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)

        return avg_loss


@manager.LOSSES.add_component
class EdgeLoss(nn.Layer):
    """
    Calculate the loss os edge.

    Args:
        aperture_size (int, optional): the  aperture size for cv2.Canny.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, aperture_size=7, ignore_index=255):
        super().__init__()
        self.aperture_size = aperture_size
        self.ignore_index = ignore_index
        self.EPS = 1e-8

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        Returns:
            (Tensor): The average loss.
        """
        # check
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert logit.shape[1] == 2, "The channel of logit should be 2."
        assert label.ndim == 3, "The ndim of label should be 3."
        mask = paddle.cast(label != self.ignore_index, 'float32')

        # obtain edge as new label
        edge = []
        label_np = (label * mask).astype("uint8").numpy()
        for i in range(label_np.shape[0]):
            label_tmp = label_np[i]
            label_tmp[label_tmp > 0] = 255
            assert label_tmp.ndim == 2

            t_lower = 50
            t_upper = 150
            edge_tmp = cv2.Canny(
                label_tmp, t_lower, t_upper, apertureSize=self.aperture_size)
            edge_tmp[edge_tmp > 0] = 1
            edge_tmp = edge_tmp[np.newaxis, ...]
            edge_tmp = paddle.to_tensor(edge_tmp, dtype="int32")
            edge.append(edge_tmp)

        edge = paddle.concat(edge, axis=0)
        assert edge.ndim == 3
        '''
        for i in range(label_np.shape[0]):
            img_e = edge.numpy().astype("uint8")[i]
            img_e[img_e > 0] = 255
            cv2.imwrite("img_e_{}.jpg".format(i), img_e)
            img_l = label_np.astype("uint8")[i]
            img_l[img_l > 0] = 255
            cv2.imwrite("img_l_{}.jpg".format(i), img_l)
        '''

        edge = F.one_hot(edge, logit.shape[1])  # N,H,W,C
        edge = paddle.cast(edge, logit.dtype)

        # loss
        logit = paddle.transpose(logit, [0, 2, 3, 1])
        loss = F.sigmoid_focal_loss(
            logit=logit, label=edge, alpha=0.25, gamma=2.0, reduction='none')
        loss = loss * mask
        avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        return avg_loss
