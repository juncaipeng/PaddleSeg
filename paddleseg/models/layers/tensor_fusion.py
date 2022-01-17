# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers


def check_shape(x, y):
    assert x.ndim == 4 and y.ndim == 4
    x_h, x_w = x.shape[2:]
    y_h, y_w = y.shape[2:]
    assert x_h >= y_h and x_w >= y_w


class FusionBase(nn.Layer):
    """Fuse two tensors. x is bigger tensor, y is smaller tensor."""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def forward(self, x, y):
        pass


class FusionAdd(FusionBase):
    """Add two tensor"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionCat(FusionBase):
    """Concat two tensor"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_out = layers.ConvBNReLU(
            2 * y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)
        out = self.conv_out(xy)
        return out


class FusionChAtten(FusionBase):
    """Use Channel attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
        if self.training:
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        atten = paddle.concat([xy_avg_pool, xy_max_pool], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionSpAtten(FusionBase):
    """Use spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten(FusionBase):
    """Combine channel and spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.ch_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def get_atten(self, xy):
        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_avg_pool = paddle.mean(xy, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        return (ch_atten, sp_atten)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten * ch_atten)
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_1(FusionChSpAtten):
    """atten = F.sigmoid(self.conv_atten(sp_atten * ch_atten))"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(self.conv_atten(sp_atten * ch_atten))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_2(FusionChSpAtten):
    """out = ch_atten * x + (1 - ch_atten) * y_up + sp_atten * x + (
            1 - sp_atten) * y_up"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)
        ch_atten = F.sigmoid(ch_atten)
        sp_atten = F.sigmoid(sp_atten)

        out = ch_atten * x + (1 - ch_atten) * y_up + sp_atten * x + (
            1 - sp_atten) * y_up
        out = self.conv_out(out)
        return out


class FusionChSpAtten_3(FusionChSpAtten):
    """atten = F.sigmoid(sp_atten + ch_atten)"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten + ch_atten)
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_4(FusionChSpAtten):
    """atten = F.sigmoid(self.conv_atten(sp_atten + ch_atten))"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(self.conv_atten(sp_atten + ch_atten))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionConvAtten(FusionBase):
    """Obtain W by two conv"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * y_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBN(
                y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False))

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        atten = F.sigmoid(self.conv_atten(xy))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionSF(FusionBase):
    """The fusion in SFNet"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.down_x = nn.Conv2D(y_ch, y_ch // 2, 1, bias_attr=False)
        self.down_y = nn.Conv2D(y_ch, y_ch // 2, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            y_ch, 2, kernel_size=3, padding=1, bias_attr=False)

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)
        norm = size[::-1].reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([size[0]]).transpose([1, 0])
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        x_size = paddle.shape(x)[2:]

        x_flow = self.down_x(x)
        y_flow = self.down_y(y)
        y_flow = F.interpolate(
            y_flow, size=x_size, mode=self.resize_mode, align_corners=False)
        flow = self.flow_make(paddle.concat([x_flow, y_flow], 1))
        y_refine = self.flow_warp(y, flow, size=x_size)

        out = x + y_refine
        out = self.conv_out(out)
        return out


class FusionSFChSpAtten(FusionSF):
    """The fusion in SFNet + combine channel and spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.ch_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def get_atten(self, xy):
        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_avg_pool = paddle.mean(xy, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        return (ch_atten, sp_atten)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        x_size = paddle.shape(x)[2:]

        x_flow = self.down_x(x)
        y_flow = self.down_y(y)
        y_flow = F.interpolate(
            y_flow, size=x_size, mode=self.resize_mode, align_corners=False)
        flow = self.flow_make(paddle.concat([x_flow, y_flow], 1))
        y_refine = self.flow_warp(y, flow, size=x_size)

        xy = paddle.concat([x, y_refine], axis=1)
        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten * ch_atten)
        out = x * atten + y_refine * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionBaseV1(nn.Layer):
    """Fuse two tensors. x is bigger tensor, y is smaller tensor."""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        if isinstance(x_chs, int):
            self.x_num = 1
            x_ch = x_chs
        elif len(x_chs) == 1:
            self.x_num = 1
            x_ch = x_chs[0]
        else:
            self.x_num = len(x_chs)
            x_num = len(x_chs)
            x_ch = x_chs[0]
            assert all([x_ch == ch for ch in x_chs]), \
                "All value in x_chs should be equal"
            assert x_ch % x_num == 0, \
                "x_ch ({}) should be the multiple of x_num ({})".format(x_ch, x_num)

            self.x_reduction = nn.LayerList()
            for _ in range(len(x_chs)):
                reduction = layers.ConvBNReLU(
                    x_ch, x_ch // x_num, kernel_size=1, bias_attr=False)
                self.x_reduction.append(reduction)

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def forward(self, xs, y):
        # check num
        x_num = 1 if not isinstance(xs, (list, tuple)) else len(xs)
        assert x_num == self.x_num, \
            "The nums of xs ({}) should be equal to {}".format(x_num, self.x_num)

        # check shape
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

        # x reduction
        if x_num == 1:
            return x
        else:
            x_list = []
            for i in range(x_num):
                x_list.append(self.x_reduction[i](xs[i]))
            x = paddle.concat(x_list, axis=1)
            return x


class FusionAddV1(FusionBaseV1):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def forward(self, xs, y):
        x = super(FusionAddV1, self).forward(xs, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionSpAttenV1(FusionBaseV1):
    """Use spatial attention"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, xs, y):
        x = super(FusionSpAttenV1, self).forward(xs, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionBaseV2(nn.Layer):
    """
    Fuse two tensors. xs are several bigger tensors, y is smaller tensor.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        if isinstance(x_chs, int):
            self.x_num = 1
            x_ch = x_chs
        elif len(x_chs) == 1:
            self.x_num = 1
            x_ch = x_chs[0]
        else:
            self.x_num = len(x_chs)
            x_ch = x_chs[0]
            assert all([x_ch == ch for ch in x_chs]), \
                "All value in x_chs should be equal"

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def check_shape(self, xs, y):
        # check num
        x_num = 1 if not isinstance(xs, (list, tuple)) else len(xs)
        assert x_num == self.x_num, \
            "The nums of xs ({}) should be equal to {}".format(x_num, self.x_num)

        # check shape
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def x_reduction(self, xs):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x += xs[i]

        return x

    def forward(self, xs, y):

        self.check_shape(xs, y)

        x = self.x_reduction(xs)
        x = self.conv_x(x)

        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)

        return x, y_up


class FusionAddV2(FusionBaseV2):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def forward(self, xs, y):
        x, y_up = super(FusionAddV2, self).forward(xs, y)

        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionWeightedAddV2(FusionBaseV2):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        alpha = self.create_parameter([self.x_num])
        self.add_parameter("alpha", alpha)

    def x_reduction(self, xs):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha[0] * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.alpha[i] * xs[i]

        return x

    def forward(self, xs, y):
        x, y_up = super(FusionWeightedAddV2, self).forward(xs, y)

        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionSpAttenV2(FusionBaseV2):
    """Use spatial attention"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, xs, y):
        x, y_up = super(FusionSpAttenV2, self).forward(xs, y)
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_Add(nn.Layer):
    """
    Fuse two tensors. xs are several bigger tensors, y is smaller tensor.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        if isinstance(x_chs, int):
            self.x_num = 1
            x_ch = x_chs
        elif len(x_chs) == 1:
            self.x_num = 1
            x_ch = x_chs[0]
        else:
            self.x_num = len(x_chs)
            x_ch = x_chs[0]
            assert all([x_ch == ch for ch in x_chs]), \
                "All value in x_chs should be equal"

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def check(self, xs, y):
        # check num
        x_num = 1 if not isinstance(xs, (list, tuple)) else len(xs)
        assert x_num == self.x_num, \
            "The nums of xs ({}) should be equal to {}".format(x_num, self.x_num)

        # check shape
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, xs, y):
        x = self.prepare_x(xs, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x += xs[i]

        x = self.conv_x(x)
        return x
        '''
        # (TODO)use sum will add scale op to infer model, check the speed by trt
        if not isinstance(xs, (list, tuple)):
            x = xs
        else:
            x = sum(xs)
        x = self.conv_x(x)
        return x
        '''

    def prepare_y(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[-1]
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, xs, y):
        self.check(xs, y)
        x, y = self.prepare(xs, y)
        out = self.fuse(x, y)
        return out


class ARM_WeightedAdd_Add(ARM_Add_Add):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        alpha = self.create_parameter([self.x_num])
        self.add_parameter("alpha", alpha)

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha[0] * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.alpha[i] * xs[i]

        x = self.conv_x(x)
        return x


class ARM_SEAdd1_Add(ARM_Add_Add):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def sp_atten(self, input):
        if self.training:
            avg_pool = F.adaptive_avg_pool2d(input, 1)
        else:
            avg_pool = paddle.mean(input, axis=[2, 3], keepdim=True)
        atten = F.sigmoid(avg_pool)
        return atten * input

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.sp_atten(x)

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.sp_atten(xs[i])

        x = self.conv_x(x)
        return x


def get_pure_tensor(x):
    if not isinstance(x, (list, tuple)):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        return paddle.concat(x, axis=1)


def avg_reduce_hw(x):
    # Reduce hw by avg
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return paddle.concat(res, axis=1)


def avg_reduce_hw_slow(x):
    # Reduce hw by avg
    x = get_pure_tensor(x)
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    return avg_pool


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when axis=[2, 3], the paddle.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = paddle.max(x, axis=[2, 3], keepdim=True)

    if use_concat:
        res = paddle.concat([avg_pool, max_pool], axis=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_hw_helper(xi, is_training, False))
        return paddle.concat(res, axis=1)


def avg_max_reduce_hw_slow(x, is_training):
    # Reduce hw by avg and max
    x = get_pure_tensor(x)

    avg_pool = F.adaptive_avg_pool2d(x, 1)
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = paddle.max(x, axis=[2, 3], keepdim=True)

    res = paddle.concat([avg_pool, max_pool], axis=1)
    return res


def avg_reduce_channel(x):
    # Reduce channel by avg
    if not isinstance(x, (list, tuple)):
        return paddle.mean(x, axis=1, keepdim=True)
    elif len(x) == 1:
        return paddle.mean(x[0], axis=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(paddle.mean(xi, axis=1, keepdim=True))
        return paddle.concat(res, axis=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = paddle.mean(x, axis=1, keepdim=True)
    max_value = paddle.max(x, axis=1, keepdim=True)

    if use_concat:
        res = paddle.concat([mean_value, max_value], axis=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return paddle.concat(res, axis=1)


class ARM_ChAttenAdd0_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(
            sum(x_chs), x_chs[0], kernel_size=1, bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_reduce_hw(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_ChAttenAdd1_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(
            2 * sum(x_chs), x_chs[0], kernel_size=1, bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_max_reduce_hw(xs, self.training)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_SpAttenAdd0_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_reduce_channel(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_SpAttenAdd1_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(
            4, 1, kernel_size=3, padding=1, bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_max_reduce_channel(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_Add_ChAttenAdd0(ARM_Add_Add):
    """
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(
            2 * y_ch, y_ch, kernel_size=1, bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd0(ARM_Add_Add):
    """
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_ChAttenAdd0_ChAttenAdd0(ARM_ChAttenAdd0_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(
            2 * y_ch, y_ch, kernel_size=1, bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_ChAttenAdd0_SpAttenAdd0(ARM_ChAttenAdd0_Add):
    """
    The length of x_chs and xs should be 2.
    """

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out
