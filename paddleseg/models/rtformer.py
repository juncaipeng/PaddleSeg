# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.backbones.transformer_utils import (
    trunc_normal_, kaiming_normal_, constant_, DropPath, Identity)


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           bias_attr=False,
           **kwargs):
    return nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias_attr=bias_attr,
        **kwargs)


def bn2d(in_channels, bn_mom=0.1, **kwargs):
    return nn.BatchNorm2D(in_channels, momentum=bn_mom, **kwargs)


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out if self.no_relu else self.relu(out)


class MLP(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = bn2d(in_channels, epsilon=1e-06)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_(m.weight, 1.0)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=1,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.same_in_out_chs = self.in_channels == self.out_channels

        self.norm = bn2d(in_channels)
        if not use_cross_kv:
            self.k = self.create_parameter(
                shape=(inter_channels, in_channels, 1, 1),
                default_initializer=paddle.nn.initializer.Normal(std=0.001))
            self.v = self.create_parameter(
                shape=(out_channels, inter_channels, 1, 1),
                default_initializer=paddle.nn.initializer.Normal(std=0.001))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                constant_(m.bias, 0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_(m.weight, 1.)
            constant_(m.bias, .0)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                constant_(m.bias, 0.)

    def _act_sn(self, x, B, H, W):
        x_shape = paddle.shape(x)
        x = x.reshape([B, self.inter_channels, H, W]) * (self.inter_channels
                                                         **-0.5)
        x = F.softmax(x, axis=1)
        x = x.reshape(x_shape)
        return x

    def _act_dn(self, x, B, H, W):
        x_shape = paddle.shape(x)
        x = x.reshape(
            [B, self.num_heads, self.inter_channels // self.num_heads, H * W])
        x = F.softmax(x, axis=3)
        x = x / (paddle.sum(x, axis=2, keepdim=True) + 1e-06)
        x = x.reshape(x_shape)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        B = x.shape[0]
        x = self.norm(x)
        if self.use_cross_kv:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should no be None when use_cross_kv"
            k, v = cross_k, cross_v
            x = x.reshape([1, -1, x.shape[2], x.shape[3]])
            x = F.conv2d(
                x,
                k,
                bias=None,
                stride=2 if not self.same_in_out_chs else 1,
                padding=0,
                groups=B)
            H, W = x.shape[2:]
            x = self._act_sn(x, B, H, W)
            x = F.conv2d(x, v, bias=None, stride=1, padding=0, groups=B)
            x = x.reshape([B, -1, H, W])
        else:
            k, v = self.k, self.v
            x = F.conv2d(
                x,
                k,
                bias=None,
                stride=2 if not self.same_in_out_chs else 1,
                padding=0)
            H, W = x.shape[2:]
            x = self._act_dn(x, B, H, W)
            x = F.conv2d(x, v, bias=None, stride=1, padding=0)

        return x


class EABlock(nn.Layer):
    """
    The EABlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_injection (bool, optional): Whether inject the high feature into low feature. Default: True
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: False
        attn_high_inter_channels (int, optional): The inter channels of atten_high module. Default: 144
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=False,
                 attn_high_inter_channels=144):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        # attn
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_channels_l),
                conv2d(in_channels_l, out_channels_l, 1, 2, 0))
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_h = ExternalAttention(
            in_channels_h,
            out_channels_h,
            inter_channels=attn_high_inter_channels,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv)
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False)
        # mlp
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else Identity()
        # fusion
        self.compression = nn.Sequential(
            bn2d(out_channels_l),
            nn.ReLU(),
            conv2d(
                out_channels_l, out_channels_h, kernel_size=1))
        self.compression.apply(self._init_weights_kaiming)
        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_channels_h),
                nn.ReLU(),
                conv2d(
                    out_channels_h,
                    out_channels_l // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                bn2d(out_channels_l // 2),
                nn.ReLU(),
                conv2d(
                    out_channels_l // 2,
                    out_channels_l,
                    kernel_size=3,
                    stride=2,
                    padding=1), )
            self.down.apply(self._init_weights_kaiming)
        # cross attention
        if use_cross_kv:
            self.cross_size = 12
            self.cross_kv = nn.Sequential(
                bn2d(out_channels_l),
                nn.AdaptiveMaxPool2D(output_size=(self.cross_size,
                                                  self.cross_size)),
                conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0))
            self.cross_kv.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_(m.weight, 1.0)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                constant_(m.bias, 0)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_(m.weight, 1.0)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)

    def forward(self, x):
        x_h, x_l = x

        # low resolution branch
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))

        # compression
        x_h_shape = paddle.shape(x_h)[2:]
        x_l_cp = self.compression(x_l)
        x_h += F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution branch
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))
        else:
            ch_h = paddle.shape(x_h)[1]
            cross_kv = self.cross_kv(x_l)
            cross_k, cross_v = paddle.split(
                cross_kv.reshape([cross_kv.shape[0], cross_kv.shape[1], -1]),
                2,
                axis=1)
            cross_k = cross_k.transpose([0, 2, 1]).reshape([-1, ch_h, 1, 1])
            cross_v = cross_v.reshape(
                [-1, self.cross_size * self.cross_size, 1, 1])

            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Layer):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=5, stride=2, padding=2, exclusive=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, inter_channels, kernel_size=1), )
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=9, stride=4, padding=4, exclusive=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, inter_channels, kernel_size=1), )
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=17, stride=8, padding=8, exclusive=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, inter_channels, kernel_size=1), )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, inter_channels, kernel_size=1), )
        self.scale0 = nn.Sequential(
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, inter_channels, kernel_size=1), )
        self.process1 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(
                inter_channels, inter_channels, kernel_size=3, padding=1), )
        self.process2 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(
                inter_channels, inter_channels, kernel_size=3, padding=1), )
        self.process3 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(
                inter_channels, inter_channels, kernel_size=3, padding=1), )
        self.process4 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(
                inter_channels, inter_channels, kernel_size=3, padding=1), )
        self.compression = nn.Sequential(
            bn2d(inter_channels * 5),
            nn.ReLU(),
            conv2d(
                inter_channels * 5,
                out_channels,
                kernel_size=1, ), )
        self.shortcut = nn.Sequential(
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(
                in_channels, out_channels, kernel_size=1), )

    def forward(self, x):
        x_shape = paddle.shape(x)[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=x_shape, mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=x_shape, mode='bilinear') + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=x_shape, mode='bilinear') + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x), size=x_shape, mode='bilinear') + x_list[3])))

        out = self.compression(paddle.concat(x_list, axis=1)) + self.shortcut(x)
        return out


class SegHead(nn.Layer):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.bn1 = bn2d(in_channels)
        self.conv1 = conv2d(
            in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn2 = bn2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(
            inter_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


@manager.MODELS.add_component
class RTFormer(nn.Layer):
    """
    The RTFormer implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        num_heads (int, optional): The num of heads in EABlock. Default: 8
        head_channels (int, optional): The channels of head in EABlock. Default: 128
        drop_rate (float, optional): The drop rate in EABlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_aux_head (bool, optional): Whether use auxiliary head. Default: True
        use_injection (list[boo], optional): Whether use injection in layer 4 and 5.
            Default: [True, True]
        in_channels (int, optional): The channels of input image. Default: 3
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_head=True,
                 use_injection=[True, True],
                 in_channels=3,
                 pretrained=None):
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(),
            nn.Conv2D(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(), )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2,
                                        1)
        self.compression3 = nn.Sequential(
            bn2d(base_chs * 4),
            nn.ReLU(),
            conv2d(
                base_chs * 4, base_chs * 2, kernel_size=1), )
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True)
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True)

        self.spp = DAPPM(base_chs * 8, spp_channels, base_chs * 2)
        self.seghead = SegHead(base_chs * 4,
                               int(head_channels * 2), num_classes)
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.seghead_extra = SegHead(base_chs * 2, head_channels,
                                         num_classes)

        self.pretrained = pretrained
        self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_(m.weight, 1.0)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.layer3_.apply(self._init_weights_kaiming)
        self.compression3.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)
        self.seghead.apply(self._init_weights_kaiming)
        if self.use_aux_head:
            self.seghead_extra.apply(self._init_weights_kaiming)

        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride),
                bn2d(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        debug = False
        if debug:
            import numpy as np
            np.random.seed(0)
            x = np.random.rand(1, 3, 512, 512).astype("float32")
            x = paddle.to_tensor(x)
            print(paddle.mean(x))

        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(
            self.compression3(x3), size=paddle.shape(x2)[2:], mode='bilinear')
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4(
            [self.relu(x3_), self.relu(x3)])  # 2c, 1/8; 8c, 1/16
        x5_, x5 = self.layer5(
            [self.relu(x4_), self.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(
            x6, size=paddle.shape(x5_)[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(paddle.concat([x5_, x6], axis=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.use_aux_head:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        if debug:
            x_out_mean = paddle.mean(x_out).numpy()
            x_out_extra_mean = paddle.mean(x_out_extra).numpy()
            print('out', x_out_mean)
            print('out', x_out_extra_mean)
            if self.base_channels == 32:
                assert np.isclose(x_out_mean, -11.117491, 0, 1e-3)
                assert np.isclose(x_out_extra_mean, -7.9465437, 0, 1e-3)
            elif self.base_channels == 64:
                assert np.isclose(x_out_mean, -21.306246, 0, 1e-3)
                assert np.isclose(x_out_extra_mean, -17.274418, 0, 1e-3)
            exit()

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list
        ]

        return logit_list
