
# uper_head_77.py

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM

# uper_head_final.py
class GlobalContextBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 inplanes,
                 inplace=False,
                 ratio=1 / 4.,
                 pooling_type='att',
                 fusion_types=('channel_add', ),):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace)

        # self.conv_hori1 = nn.Conv1d(in_channels, out_channels, kernel_size=img_size, stride=img_size, padding=3)
        # self.conv_verti1 = nn.Conv1d(in_channels, out_channels, kernel_size=img_size, stride=img_size, padding=3)

        self.conv_hori1 = nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.conv_verti1 = nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)

        self.conv_77 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)

        self.conv_55_1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.conv_55_2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)

        self.conv_33 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_55 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)


        self.conv_33_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_33_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_33_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # self.convs = ConvModule(
        #     self.in_channels,
        #     self.out_channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg,
        #     inplace=False)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, self.img_size, self.img_size]),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, self.img_size, self.img_size]),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        # print("X_SIZE", x.size())
        # print("img_size",self.img_size)
        # print("self.in_channels", self.in_channels)
        # print("self.out_channels", self.out_channels)
        # print("self.inplanes", self.inplanes)

        # if self.pooling_type == 'att':
        #
        #     x_h = x
        #     x_v = x
        #     # [N, C, H * W]
        #     x_h = x_h.view(batch, channel, height * width)
        #     # print('input_x.size',x_h.size())
        #     # [2 ,512, H*W]
        #     x_hori = self.conv_hori1(x_h)
        #     # print('input_x_hori1.size', x_hori.size())
        #     # [2, 512, H]
        #     # [N, 1, C, H * W]
        #     x_hori = x_hori.unsqueeze(3)
        #     x_hori = x_hori.permute(0, 1, 3, 2)
        #     # print('x_hori.size', x_hori.size())
        #
        #     x_v = torch.rot90(x_v, k=-1, dims=[2, 3])
        #     # print('x_v', x_v.size())
        #     x_v = x_v.contiguous().view(batch, channel, height * width)
        #     x_verti = self.conv_verti1(x_v)
        #     x_verti = x_verti.unsqueeze(3)
        #     # print('x_verti.size', x_verti.size())
        #
        #     # context = torch.matmul(x_verti, x_hori) # 128 * 128
        #     context = torch.matmul(x_hori, x_verti) # 1 * 1
        #
        #     # print('context.size', context.size())
        #     #  context = torch.sigmoid(context)

        # x_1d = x
        # x_1d = x_1d.view(batch, channel, height * width)
        # x_1d = self.conv_hori1(x_1d)
        # x_2d = x_1d.view(batch, channel, height, width)
        # x_2d = torch.rot90(x_2d, k=-1, dims=[2, 3])
        # x_1d = x_2d.contiguous().view(batch, channel, height * width)
        # x_1d = self.conv_verti1(x_1d)
        # context = x_1d.view(batch, channel, height, width)

        # 7*7
        context = self.conv_77(x)

        # 5*5
        # context = self.conv_55_1(x)
        # context = self.conv_55_2(context)

        # # 3*3 5*5
        # context = self.conv_33(x)
        # context = self.conv_55(context)

        # # # 3*3
        # context = self.conv_33_1(x)
        # context = self.conv_33_2(context)
        # context = self.conv_33_3(context)


        # if self.pooling_type == 'att':
        #     # [N, C, H ,W]
        #     input_x = x
        #     # [N, C, H * W]
        #     input_x = input_x.view(batch, channel, height * width)
        #     # [N, 1, C, H * W]
        #     input_x = input_x.unsqueeze(1)
        #     # [N, 1, H, W]
        #     context_mask = self.conv_mask(x)
        #     # [N, 1, H * W]
        #     context_mask = context_mask.view(batch, 1, height * width)
        #     # [N, 1, H * W]
        #     context_mask = self.softmax(context_mask)
        #     # [N, 1, H * W, 1]
        #     context_mask = context_mask.unsqueeze(-1)
        #     # [N, 1, C, 1]
        #     context = torch.matmul(input_x, context_mask)
        #     # [N, C, 1, 1]
        #     context = context.view(batch, channel, 1, 1)
        # else:
        #     # [N, C, 1, 1]
        #     context = self.avg_pool(x)


        # print('context_size', context.size())

        return context

    def forward(self, x):
        # x = self.convs(x)

        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # print('out1', out.size())
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            # print("context", context.size())
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

            # out = out + context

            out = self.conv1(out)
            # print('out2', out.size())
            # print("out.shape",out.shape)


        return out


@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            # print(in_channels)
         #   l_conv = ConvModule(
          #      in_channels,
           #     self.channels,
            #    1,
            #    conv_cfg=self.conv_cfg,
            #    norm_cfg=self.norm_cfg,
             #   act_cfg=self.act_cfg,
              #  inplace=False)
            l_conv = GlobalContextBlock(
                in_channels,
                self.channels,
                1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplanes=in_channels,
                inplace=False)
            self.lateral_convs.append(l_conv)
            #fpn_conv = GlobalContextBlock(
               # self.channels,
                #self.channels,
             #   3,
              #  padding=1,
              #  conv_cfg=self.conv_cfg,
              #  norm_cfg=self.norm_cfg,
              #  act_cfg=self.act_cfg,
               # inplanes=self.channels,
              #  inplace=False)

            fpn_conv = ConvModule(
                 self.channels,
                 self.channels,
                 3,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg,
                 inplace=False)

            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # # self self.lateral_convs
        # print('self.lateral_convs', self.lateral_convs)
        # # self self.fpn_convs
        # print('self.fpn_convs', self.fpn_convs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        # print("output", output.shape)
        return output

