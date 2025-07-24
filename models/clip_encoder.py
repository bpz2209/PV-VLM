# Copyright (c) 2023-2024 DeepSeek.
#
# 本软件按照MIT许可发布，允许任何人免费使用、复制、修改、合并、发布、分发、再许可和/或销售软件的拷贝，
# 并允许被提供软件的人这样做，但必须保留以上版权声明和许可声明。
# 软件是按“现状”提供，不包含任何明示或暗示的担保，作者或版权持有人不承担任何因软件使用而产生的责任。

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms
from einops import rearrange
from transformers import AutoConfig, AutoModel
# 导入创建SigLIP ViT模型的函数
from models.siglip_vit import create_siglip_vit


class CLIPVisionTower(nn.Module):
    """
    这是一个视觉特征提取器类，用于从图像中提取特征，支持多种模型：
    1. SigLIP ViT模型
    2. SAM ViT模型（假设）
    3. HuggingFace的CLIPVisionModel

    该模型可以处理图像归一化、模型构建、前向传播及特征选择。    
    """

    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        patch: int = 16,
        select_feature: str = "patch",    # 特征选择方式，默认选择patch tokens
        select_layer: int = -2,             # 从哪个层次选择特征（负数表示倒数第几层）
        select_layers: list = None,         # 如果需要选择多个层，可以在此指定
        ckpt_path: str = "",                # 预训练模型的路径
        pixel_mean: Optional[List[float]] = None,  # 图像归一化的均值
        pixel_std: Optional[List[float]] = None,   # 图像归一化的方差
        **kwargs,
    ):
        # 调用父类nn.Module的构造函数
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature     # 存储特征选择模式
        self.select_layer = select_layer         # 存储要选择的层
        self.select_layers = select_layers       # 如果需要，存储多个层的索引

        # 构造vision_tower模型所需的参数字典
        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
            "patch": patch,
        }
        # 将其它额外参数加入到字典中
        vision_tower_params.update(kwargs)
        # 根据参数构造视觉模型并获取前向传播时需要额外传递的参数
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params
        )

        # 如果提供了图像归一化的参数，则初始化归一化变换，否则不使用归一化
        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        """
        根据模型名称判断并构造不同的视觉tower模型：
        - 如果模型名称以'siglip'开头，则使用siglip_vit
        - 如果模型名称以'sam'开头，则使用sam_vit（需自己实现或导入）
        - 否则，使用Transformer库中的CLIPVisionModel
        
        返回构造好的模型和前向传播时额外需要传递的关键字参数。
        """
        # if self.model_name.startswith("siglip"):
        #     # 对于siglip模型，保持select_feature为"same"
        #     self.select_feature = "same"
        #     # vision_tower = create_siglip_vit(**vision_tower_params)
        #     # forward_kwargs = dict()
        #     from transformers import AutoModel
        #     vision_tower = AutoModel.from_pretrained(**vision_tower_params)
        #     forward_kwargs = dict(output_hidden_states=True)

        # elif self.model_name.startswith("sam"):
        #     # 如果模型名称以'sam'开头，调用相应的构造函数（需保证create_sam_vit已定义）
        #     vision_tower = create_sam_vit(**vision_tower_params)
        #     forward_kwargs = dict()

        # else:  # 使用HuggingFace提供的CLIPVisionModel
        #     from transformers import CLIPVisionModel

        #     vision_tower = CLIPVisionModel.from_pretrained(**vision_tower_params)
        #     # CLIPVisionModel前向传播需要输出所有隐藏层，因此设置output_hidden_states参数为True
        #     forward_kwargs = dict(output_hidden_states=True)

        try:
            self.config = AutoConfig.from_pretrained(vision_tower_params["ckpt"])
        except Exception as e:
            print("Error loading config from model_name")
        self.config.vision_config.image_size = vision_tower_params["image_size"]
        try:
            vision_tower = AutoModel.from_pretrained(
                vision_tower_params["ckpt"],
                trust_remote_code=True,
                local_files_only=True,
                ignore_mismatched_sizes=True,
                config=self.config,
            )
        except Exception as e:
            vision_tower = AutoModel.from_pretrained(
                vision_tower_params["ckpt"],
                ignore_mismatched_sizes=True,
                config=self.config,
            )
        forward_kwargs = dict()
        if vision_tower_params['model_name'].startswith("siglip"):
            self.select_feature = "same"
        elif vision_tower_params['model_name'].startswith("clip"):
            forward_kwargs = dict(output_hidden_states=True)
        
        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        """
        根据设定的select_feature参数选择输出的特征：
        - 如果输出类型为Tensor，则认为已经是选择好的层数据
        - 如果输出为包含多个隐藏状态的对象，则从中选择指定的层
        针对不同的特征选择策略：
            - "patch": 去除CLS token，只保留patch tokens
            - "cls_patch"或"same": 不做额外处理，返回所有Token
        """
        # 检查forward的输出是否为Tensor类型
        if isinstance(image_forward_outs, torch.Tensor):
            # 如果是Tensor，说明输出已经为所需要的层特征
            image_features = image_forward_outs
        else:
            # 否则，提取指定的隐藏层（select_layer支持负索引）
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            # 如果选择模式为"patch"，则假定第一个Token为CLS，故去除第一个Token
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            # 如果选择模式为"cls_patch"，则保留所有Token（包括CLS token）
            image_features = image_features
        elif self.select_feature == "same":
            # "same"模式下不做任何修改
            image_features = image_features
        else:
            # 如果传入未识别的选择模式，则抛出异常
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        """
        前向传播函数：
        - 对输入图像执行归一化（如果已配置）
        - 使用视觉tower模型计算图像特征
        - 根据select_feature选择合适的特征层返回

        参数:
            images (torch.Tensor): 形状为 [batch_size, 3, H, W] 的图像张量

        返回:
            image_features (torch.Tensor): 形状为 [batch_size, n_patch, d] 的特征张量
        """
        # 如果配置了归一化，则先归一化图像
        if self.image_norm is not None:
            images = self.image_norm(images)
    
        # 使用视觉tower模型计算图像特征，forward_kwargs可能包含多余的参数
        image_forward_outs = self.vision_tower.vision_model(images, output_hidden_states=True)
        print("shape of image_forward_outs:", image_forward_outs.pooler_output.shape, "shape of image_forward_outs.last_hidden_state:", image_forward_outs.last_hidden_state.shape)
        # 根据设置选择需要的特征层
        # image_features = self.feature_select(image_forward_outs)
        return image_forward_outs.pooler_output, image_forward_outs.last_hidden_state
