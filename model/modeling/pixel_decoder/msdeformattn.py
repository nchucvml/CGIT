# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn
from torch.nn.modules.module import Module

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)

        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        # print("encoder_layerencoder_layer")
        # print(self.encoder)
        # print("encoder_layerencoder_layer")
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)

            # print(spatial_shape) # (64, 64) # (128, 128) # (256, 256)

            # print(src.shape)
            # torch.Size([1, 256, 64, 64])
            # torch.Size([1, 256, 128, 128])
            # torch.Size([1, 256, 256, 256])
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)
        return memory, spatial_shapes, level_start_index

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

class my_AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3, 1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                             self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3, 1).contiguous()

        return input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', ceil_mode=' + str(self.ceil_mode) \
               + ', count_include_pad=' + str(self.count_include_pad) + ')'


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            transformer_dropout: float,
            transformer_nheads: int,
            transformer_dim_feedforward: int,
            transformer_enc_layers: int,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            # deformable transformer encoder args
            transformer_in_features: List[str],
            common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        transformer_in_features2 = ['res5']
        transformer_input_shape2 = {
            k: v for k, v in input_shape.items() if k in transformer_in_features2
        }
        """
        print("transformer_input_shape2:")
        print(transformer_input_shape2)
        print("transformer_in_features:")
        print(transformer_in_features) # v2: ['res2', 'res3', 'res4']
        print("transformer_input_shape:")
        print(transformer_input_shape) #v2:{'res2': ShapeSpec(channels=128, height=None, width=None, stride=4), 'res3': ShapeSpec(channels=256, height=None, width=None, stride=8), 'res4': ShapeSpec(channels=512, height=None, width=None, stride=16)}
        """
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        #0.0# print("input_shape: ",input_shape)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        #0.0#print("self.in_features: ",self.in_features) #res2.3.4.5  ['res2', 'res3', 'res4', 'res5']
        #0.0#print("type(self.in_features):")
        #0.0#print(type(self.in_features))
        self.feature_strides = [v.stride for k, v in input_shape]
        #0.1#print("self.feature_strides: ", self.feature_strides) # [4, 8, 16, 32]
        self.feature_channels = [v.channels for k, v in input_shape]
        #0.1#print("self.feature_channels: ", self.feature_channels) # [128, 256, 512, 1024]

        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        transformer_input_shape2 = sorted(transformer_input_shape2.items(), key=lambda x: x[1].stride)
        #transformer_input_shape2 = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        #0.0#print(" transformer_input_shape: ", transformer_input_shape)
        #0.0# print(" transformer_input_shape2: ", transformer_input_shape2)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        self.transformer_in_features2 = [k for k, v in transformer_input_shape2]
        #0.1#    print("self.transformer_in_features: ",self.transformer_in_features) #3個res2.3.4/res3.4.5
        #0.1#    print("self.transformer_in_features2: ", self.transformer_in_features2)

        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        #0.0#print("transformer_in_channels: ", transformer_in_channels)
        transformer_in_channels2 = [v.channels for k, v in transformer_input_shape2]
        #0.0#print("transformer_in_channels2: ", transformer_in_channels2)
        # 1#print(transformer_in_channels) # transformer_in_channels:[256, 512, 1024] => swin-base21K
        # 0.1 #print("transformer_in_channels: ",transformer_in_channels)   # [128, 256, 512]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers
        #0.1#        print("self.transformer_feature_strides :", self.transformer_feature_strides) #4,8,16

        self.transformer_feature_strides2 = [v.stride for k, v in transformer_input_shape2]
        #0.1#        print("self.transformer_feature_strides2 :", self.transformer_feature_strides2) #32
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        #0.1#        print("self.transformer_num_feature_levels :", self.transformer_num_feature_levels) #3
        self.transformer_num_feature_levels2 = len(self.transformer_in_features2)
        #0.1#        print("self.transformer_num_feature_levels2 :", self.transformer_num_feature_levels2) #1

        # 0.1 #        print("self.transformer_feature_strides: ",self.transformer_feature_strides)  # 8,16,32 (res3, res4, res5的結果)
        # self.transformer_feature_strides = [4, 8, 16] #註解 res3.4.5 改成 2.3.4加入
        #
        # print(self.transformer_feature_strides)
        # print(self.in_features) #res2,res3,res4,res5
        # print(self.feature_strides) #4,8,16,32
        # print(self.feature_channels) #128,256,512,1024
        #
        # print(self.transformer_in_features) #res3,res4,res5  v2:2,3,4
        # print(transformer_in_channels) #256,512,1024  v2:128,256,512
        # print(self.transformer_feature_strides) #8,16,32  v2:4,8,16
        # print(self.transformer_num_feature_levels) #3
        #
        # ========================================================================2023.02.08
        self.double2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # setting: False
        self.double4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)  # setting: False
        # self.avgpool = nn.AvgPool2d((2, 1), stride=(2, 1))
        self.ap = my_AvgPool2d((1, 2), stride=(1, 2))
        # ========================================================================2023.02.08

        # ========================================================================2023.02.10
        self.conv2d_512 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        # self.conv2d_256 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        # ========================================================================2023.02.10

        # ========================================================================2023.02.13
        self.double8 = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)
        # ========================================================================2023.02.13

        # ========================================================================2023.03.24
        self.conv2d_512_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv2d_512_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        # ========================================================================2023.03.24

        # ========================================================================2023.03.25
        # self.conv2d_512_4 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        # self.conv2d_512_5 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        # ========================================================================2023.03.25

        # ========================================================================2023.04.13
        self.conv2d_256_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.conv2d_256_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        # ========================================================================2023.04.13



        # ========================================================================2023.04.06
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0)
        # #self.conv_pool = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=1)
        # self.conv_pool = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), padding=1)
        # self.conv_d1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        # self.conv_d4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=4, dilation=4) #3,3
        # self.conv_d8 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=8, dilation=8)
        # self.conv_d16 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=16, dilation=16)
        # #srcs[idx] = torch.cat((srcs[idx], self.double2(srcs2[idx])), 1)
        # #self.concat = torch.cat((self.conv_pool,self.conv_d1,self.conv_d4,self.conv_d8,self.conv_d16),dim=1)
        # self.instance_norm = nn.InstanceNorm2d(num_features=320)
        # self.dropout = nn.Dropout2d(p=0.25)
        # print("----------------------------------------------ffffffffffffffffffdsfdsfds")
        # print("----------------------------------------------ffffffffffffffffffdsfdsfds")
        # print("----------------------------------------------ffffffffffffffffffdsfdsfds")
        # print("----------------------------------------------ffffffffffffffffffdsfdsfds")
        # self.conv2d_1280 = nn.Conv2d(1280, 256, kernel_size=1, stride=1)

        # ========================================================================2023.04.06
        if self.transformer_num_feature_levels > 1:  # =3
            input_proj_list = []
            # 0.0 #print("conv_dim:", conv_dim)
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                # 0.1 # print("in_channels: ", in_channels) #r5:1024, r4:512, r3:256   #v2: 512,256,128
                #print(transformer_in_channels[::-1]) #[1024, 512, 256]
                # 0.1 # print("transformer_in_channels[::-1]: ", transformer_in_channels[::-1])  #v2: 512,256,128
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
                # print (input_proj_list)
            self.input_proj = nn.ModuleList(input_proj_list)
            # 0.1 #  print("self.input_proj:", self.input_proj)   # channel will become same=256
            input_proj_list2 = []
            for in_channels2 in transformer_in_channels2[::-1]:
                # 0.1 # print("in_channels2: ", in_channels2)  # 1024
                # 0.1 # print(transformer_in_channels2[::-1])  # [1024]
                input_proj_list2.append(nn.Sequential(
                    nn.Conv2d(in_channels2, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
                # print (input_proj_list)
            self.input_proj = nn.ModuleList(input_proj_list)
            self.input_proj2 = nn.ModuleList(input_proj_list2)
            # 0.1 #   print("self.input_proj:", self.input_proj)  # 512,256,128 -> 256
            # 0.1 #   print("self.input_proj2:", self.input_proj2)  # 1024 -> 256

        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)  # 保持輸入和輸出的方差一致，避免所有輸出值都趨向於0。
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,  # 256
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        # print(self.transformer_in_features[::-1])  # ['res5', 'res4', 'res3']
        # for idx2, f2 in enumerate(self.transformer_in_features2[::-1]):
        #     x = features[f2].float()
        #     print("idx2, f2_shape: ", x.shape)
        srcs2 = []
        # print("----------------------------------------------ffffffffffffffffffdsfdsfds")
        for idx2, f2 in enumerate(self.transformer_in_features2[::-1]):
            y = features[f2].float()
            # 0.0 #print("f2: ", f2)
            # 0.1 #print(y.shape)  # 1, 1024, 32, 32
            srcs2.append(self.input_proj2[idx2](y))
            # 0.1 #print("srcs2[idx=0].shape: ", srcs2[idx2].shape)  # 1, 256, 32, 32
            # print("srcs2[idx=0].shape: ", srcs2[idx2].shape)
            # # = 0.1 ==================================================================2023.04.05
            # # srcs2[0] = 1,256,32,32
            # if idx2 == 0:
            #     pool = self.pool(srcs2[idx2])
            #     #print("pool.shape: ", pool.shape)
            #     pool = self.conv_pool(pool)
            #     #print("pool.shape: ", pool.shape)
            #     d1 = self.conv_d1(srcs2[idx2])
            #     #print("srcs2[idx=0].shape: ", srcs2[idx2].shape)# 1, 256, 24, 24
            #     #print("d1.shape: ", d1.shape) # 1, 64, 24, 24
            #     #y = self.concat([srcs2[idx2], d1])
            #     #srcs[idx] = torch.cat((srcs[idx], self.double4(srcs2[idx - 1])), 1)
            #     y = torch.cat((srcs2[idx2], d1), 1)
            #     #print("y.shape: ",y.shape) #1, 320, 24, 24
            #     y = F.relu(y)
            #
            #     d4 = self.conv_d4(y)
            #     #print("d4.shape: ", d4.shape)
            #     # y = self.concat([srcs2, d4])
            #     y = torch.cat((srcs2[idx2], d4), 1)
            #     #print("y.shape: ", y.shape)
            #     y = F.relu(y)
            #     d8 = self.conv_d8(y)
            #     #print("d8.shape: ", d8.shape)
            #     # y = self.concat([srcs2, d8])
            #     y = torch.cat((srcs2[idx2], d8), 1)
            #     #print("y.shape: ", y.shape)
            #     y = F.relu(y)
            #     d16 = self.conv_d16(y)
            #     #print("d16.shape: ", d16.shape)
            #
            #     srcs2[idx2]= torch.cat((pool, d1, d4, d8, d16), 1)
            #     #print("srcs2.shape: ", srcs2[idx2].shape)
            #     #srcs2 = self.concat([pool, d1, d4, d8, d16])
            #     srcs2[idx2] = self.conv2d_1280(srcs2[idx2])
            #     #print("srcs2.shape: ", srcs2[idx2].shape)
            #     #srcs2[idx2] = self.instance_norm(srcs2[idx2])
            #
            #     srcs2[idx2] = F.relu(srcs2[idx2])
            #     srcs2[idx2] = self.dropout(srcs2[idx2])

            # ========================================================================2023.04.05
            #print("srcs2[idx=0].shape: ", srcs2[idx2].shape)

        for idx, f in enumerate(self.transformer_in_features[::-1]):  # f(string): res5,res4,res3 (list) # idx = 0, 1 ,2
            # 0.0 #
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            x = features[f].float()  # deformable detr does not support half precision
            # if f == 'res5':
            #     print("f == res5 & idx == 0")
            #     x = features[f].float()
            # if f == 'res4':
            #     print("f == res4 & idx == 1")
            # if f == 'res3':
            #     print("f == res3 & idx == 2")
            # ----res4,3,2: --------------------------------------------------
            # 0.1 # print("x.shape: ",x.shape)  # [1,512,64,64],[1,256,128,128],[1,128,256,256]
            # ----res5,4,3: --------------------------------------------------
            # 0.0 # print(x.shape)  # [1,1024,32,32],[1,512,64,64],[1,256,128,128]
            # 0.0 # print("- * - * - * - * - * - * -") # 做完src.會變成統一channel
            srcs.append(self.input_proj[idx](x))

            """ res4,3,2:
            # 0.1 # 
            srcs[0].shape: [1,256,64,64]
            srcs[1].shape: [1,256,128,128]
            srcs[2].shape: [1,256,256,256]
            """

            # 0.1 #
            if idx == 0:
                # 04.11#
                srcs[idx] = torch.cat((srcs[idx], self.double2(srcs2[idx])), 1)  # 1, 512, 64, 64
            # 0.1 #
                # 04.11#
                srcs[idx] = self.conv2d_512(srcs[idx])  # 1, 256, 64, 64  # stage3_ 1x1 conv
            # # #　print("self.input_proj[idx](x).shape: ", self.input_proj[idx](x).shape)
            # = 0.1 ==================================================================2023.03.24
                srcs01 = self.double2(srcs[idx])  # 1, 256, 128, 128
            # ========================================================================2023.03.24
                # 4.12 start
                srcs03 = self.double4(srcs[idx])  # 1, 256, 256, 256
                # 4.12 end

            if idx == 1:
                # 0.1 #
                #04.11#
                srcs[idx] = torch.cat((srcs[idx], self.double4(srcs2[idx - 1])), 1)  # 1, 512, 128, 128
                # srcs[idx] = self.ap(srcs[idx])

                # 0.1 #
                # ========================================================================2023.03.30 v3
                # srcs[idx] = self.conv2d_512_2(srcs[idx])  # 1, 256, 128, 128  # stage2_ 1x1 conv
                #04.11#
                srcs[idx] = self.conv2d_512(srcs[idx])  # 1, 256, 128, 128  # stage2_ 1x1 conv
                # ========================================================================2023.03.30 v3

                # srcs[idx] = self.double2(srcs[idx-1])
                # 0.0 # print("srcs[idx==1].shape: ", srcs[idx].shape)



            # # = 0.1 ==================================================================2023.03.24
            #     srcs[idx] = torch.cat((srcs[idx], srcs01), 1)  # 1, 512, 128, 128
            #     srcs[idx] = self.conv2d_512_2(srcs[idx])  # 1, 256, 128, 128  # stage2_ 1x1 conv
            #     # srcs[idx] = self.conv2d_512_4(srcs[idx])  # 1, 256, 128, 128  # stage2_ 1x1 conv # new 2023.03.25
            #     srcs02 = self.double2(srcs[idx])  # 1, 256, 256, 256
            # # ========================================================================2023.03.24
                # 4.12 start 03.24 被 04.12 版註解
                srcs01 = torch.mul(srcs01, srcs[idx])
                srcs[idx] = torch.add(srcs01, srcs[idx])

                # 4.12 end   03.24 被 04.12 版註解
                # 4.13
                # srcs[idx] = self.conv2d_256_2(srcs[idx])
                # 4.13

            if idx == 2:
                # 0.1 #
                # 04.11 #
                srcs[idx] = torch.cat((srcs[idx], self.double8(srcs2[idx - 2])), 1)  # 1, 512, 256, 256
                # srcs[idx] = self.ap(srcs[idx])

                # 0.1 #
                # ========================================================================2023.03.30 v3
                # srcs[idx] = self.conv2d_512_3(srcs[idx])  # 1, 256, 256, 256  # stage1_ 1x1 conv
                # 04.11 #
                srcs[idx] = self.conv2d_512(srcs[idx])  # 1, 256, 256, 256  # stage1_ 1x1 conv
                # ========================================================================2023.03.30 v3

                # srcs[idx] = self.double4(srcs[idx-2])
                # 0.0 #print("srcs[idx==2].shape: ", srcs[idx].shape)

            # # = 0.1 ==================================================================2023.03.24
            #     srcs[idx] = torch.cat((srcs[idx], srcs02), 1)  # 1,512,256,256
            #     srcs[idx] = self.conv2d_512_3(srcs[idx])  # 1,256,256,256  # stage1_ 1x1 conv
            #     # srcs[idx] = self.conv2d_512_5(srcs[idx])  # 1,256,256,256  # stage1_ 1x1 conv # new 2023.03.25
            # # ========================================================================2023.03.24

                # 4.12 start 03.24 被 04.12 版註解
                srcs03 = torch.mul(srcs03, srcs[idx])
                srcs[idx] = torch.add(srcs03, srcs[idx])
                # 4.12 end   03.24 被 04.12 版註解
                # 4.13
                # srcs[idx] = self.conv2d_256_3(srcs[idx])
                # 4.13

            # self.input_proj[idx](x).shape)  # [1,256,32,32] # [1,256,64,64] # [1,256,128,128]
            pos.append(self.pe_layer(x))
            # self.pe_layer(x).shape)  # [1,256,32,32] # [1,256,64,64] # [1,256,128,128]

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        # spatial_shapes.shape: torch.Size([3, 2])

        # spatial_shapes: tensor([[32, 32],
        #                         [64, 64],
        #                         [128, 128]], device='cuda:0')

        # level_start_index: tensor([0, 1024, 5120], device='cuda:0')

        bs = y.shape[0]
        # y.shape: # torch.Size([1, 21504, 256]) # 21504 = 16384+4096+1024

        split_size_or_sections = [None] * self.transformer_num_feature_levels  # 3
        for i in range(self.transformer_num_feature_levels):  # i = 0,1,2
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
                # tensor(1024, device='cuda:0')
                # tensor(4096, device='cuda:0')
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
                # tensor(16384, device='cuda:0')

        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0

        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
            # len(out) = 3

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            # 0.0 # print(lateral_conv)   # Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
            output_conv = self.output_convs[idx]
            # 0.0 # print(output_conv)    # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            cur_fpn = lateral_conv(x)

            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)  # print(len(y))  #1
            out.append(y)  # print(len(out)) #4
        # ------------------------------------------------------------------v1
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
            # print(o.shape)
            # [1,256,32,32], [1,256,64,64], [1,256,128,128], [1,256,256,256]
        # ------------------------------------------------------------------v1
        # multi_scale_features[0].shape: torch.Size([1, 256, 32, 32])
        # multi_scale_features[1].shape: torch.Size([1, 256, 64, 64])
        # multi_scale_features[2].shape: torch.Size([1, 256, 128, 128])
        # ------------------------------------------------------------------v1
        return self.mask_features(out[-1]), out[0], multi_scale_features
