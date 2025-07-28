from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoImageProcessor
import torch
import torch.nn as nn
from torch import Tensor
from math import sqrt
from models.patchtst import PatchTST_no_Head
from models.StandNorm import Normalize
####################################
# 通用归一化层
####################################
class NormalizationLayer(nn.Module):
    def __init__(self, norm_type: str, num_features: int, eps: float = 1e-5, affine: bool = True, num_groups: int = 32):
        """
        通用归一化层，可选择 'layernorm'、'batchnorm' 或 'groupnorm'
        """
        super(NormalizationLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(num_features, eps=eps, affine=affine)
        elif norm_type == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.norm_type = norm_type

    def forward(self, x: Tensor) -> Tensor:
        # 对于 BatchNorm1d 和 GroupNorm，输入要求 [B, C, L]
        if self.norm_type in ['batchnorm', 'groupnorm']:
            if x.dim() == 3:
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
                return x
            else:
                return self.norm(x)
        else:
            return self.norm(x)
def Rotate_PositionEmbedding(d_model, max_len=5000):
    """
    生成旋转位置编码
    :param d_model: 模型的维度
    :param max_len: 最大长度
    :return: 位置编码矩阵
    """
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # 增加 batch 维度
###############################
# 交叉注意力模块
###############################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        :param embed_dim: 输入特征的维度，要求查询和键值维度相同
        :param num_heads: 多头注意力的头数
        :param dropout: dropout 概率
        """
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        :param query: [B, L_query, embed_dim]
        :param key:   [B, L_key, embed_dim]
        :param value: [B, L_key, embed_dim]
        :return: attn_output: [B, L_query, embed_dim]
                 attn_weights: [B, num_heads, L_query, L_key]
        """
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output, attn_weights
class Head(nn.Module):
    def __init__(self, nf, horizons, dropout=0.0):
        super(Head, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc = nn.Sequential(
            # nn.Linear(nf, nf),
            # nn.GELU(),
            nn.Linear(nf, horizons)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        # print("x shape:", x.shape)
        x = self.fc(x)
        x = self.dropout(x)
        return x

class PV_VLM(nn.Module):
    def __init__(self, configs):
        super(PV_VLM, self).__init__()
        self.horizons = configs.horizons
        self.input_size = configs.input_size
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_ff = configs.d_ff        # 时序特征维度
        self.d_model = configs.d_model  # 时序 patch embedding 和 reprogramming 用的维度
        self.n_heads = configs.n_heads
        self.top_k = 5 if self.horizons > 5 else self.horizons
        # 消融实验控制参数
        self.wio_ts = configs.wio_ts       # 是否使用时序信息
        self.wio_img = configs.wio_img     # 是否使用图像信息
        self.wio_prompt = configs.wio_prompt   # 是否使用 prompt 信息
        self.only_ts = configs.only_ts
        if configs.model_name == 'GPT2':
            self.repo = 'openai-community/gpt2'
        elif configs.model_name == 'BERT':
            self.repo = 'google-bert/bert-base-uncased'
        elif configs.model_name == 'Qwen2.5-0.5B':
            self.repo = 'Qwen/Qwen2.5-0.5B'
        elif configs.model_name == 'Qwen2.5-1.5B':
            self.repo = 'Qwen/Qwen2.5-1.5B'
        elif configs.model_name == 'Qwen2.5-3B':
            self.repo = 'Qwen/Qwen2.5-3B'
        try:
            self.config = AutoConfig.from_pretrained(self.repo)
        except Exception as e:
            self.config = AutoConfig.from_pretrained(self.repo)
        # self.config.num_hidden_layers = configs.num_hidden_layers
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        if configs.horizons > 20:
            self.config.n_positions = 3072
        
        try:
            self.llm_model = AutoModel.from_pretrained(
                self.repo,
                trust_remote_code=True,
                local_files_only=True,
                ignore_mismatched_sizes=True if configs.horizons > 20 else False,
                config=self.config,
            )
        except Exception as e:
            self.llm_model = AutoModel.from_pretrained(
                self.repo,
                ignore_mismatched_sizes=True if configs.horizons > 20 else False,
                config=self.config,
            )
        self.d_llm = self.llm_model.config.hidden_size

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.repo,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo)

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        try:
            self.vision_tower_config = AutoConfig.from_pretrained(configs.vision_tower_ckpt)
        except Exception as e:
            self.vision_tower_config = AutoConfig.from_pretrained(configs.vision_tower_ckpt)
        self.vision_tower_config.vision_config.image_size = configs.image_size
        try:
            self.vision_tower = AutoModel.from_pretrained(
                configs.vision_tower_ckpt,
                trust_remote_code=True,
                local_files_only=True,
                ignore_mismatched_sizes=True,
                config=self.vision_tower_config,
            )
        except Exception as e:
            self.vision_tower = AutoModel.from_pretrained(
                configs.vision_tower_ckpt,
                ignore_mismatched_sizes=True,
                config=self.vision_tower_config,
            )
        try:
            self.vision_tower_tokenizer = AutoTokenizer.from_pretrained(
                configs.vision_tower_ckpt,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            self.vision_tower_tokenizer = AutoTokenizer.from_pretrained(configs.vision_tower_ckpt)
        if self.vision_tower_tokenizer.eos_token:
            self.vision_tower_tokenizer.pad_token = self.vision_tower_tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.vision_tower_tokenizer.add_special_tokens({'pad_token': pad_token})
            self.vision_tower_tokenizer.pad_token = pad_token
        # self.vision_tower.config.text_config.max_position_embeddings = 2048
        for param in self.vision_tower.parameters():
            param.requires_grad = False

        # ---------- soft prompt ----------
        self.num_soft = configs.num_soft_prompt_text_tokens
        # 用 LLM embedding 初始化（可选）；这里用正态随机即可
        self.soft_prompt = nn.Parameter(torch.randn(self.num_soft, self.d_llm))
        self.patchtst = PatchTST_no_Head(configs)
        self.img_projection = nn.Linear(self.vision_tower.config.vision_config.hidden_size, configs.d_model)
        self.llm_out_projection = nn.Linear(self.d_llm, configs.d_model)
        self.conv = nn.Conv1d(self.d_llm, self.d_llm, 3, padding=1).to(torch.bfloat16)
        self.connector = nn.Sequential(
            nn.Linear(self.vision_tower.config.vision_config.hidden_size, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm)
        )
        self.text_projection = nn.Linear(self.vision_tower.config.text_config.hidden_size, configs.d_model)
        self.ts_projection = nn.Linear(configs.d_model, configs.d_model)
        self.normalize = Normalize('layernorm', 1)
        self.cma = CrossAttention(self.d_model, self.n_heads, configs.dropout)
        self.cma_img_ts = CrossAttention(self.d_model, self.n_heads, configs.dropout)
        self.patch_nums = int((self.input_size - self.patch_len) / self.stride + 2)
        self.head_nf = configs.d_model * self.patch_nums
        self.output_projection = Head(self.head_nf, self.horizons)
        self.input_size = configs.input_size
        self.horizons = configs.horizons
        self.description = configs.description

    def forecast(self, imgs, x):
        # 时序数据预处理
        B, T, N = x.size()
        x_enc = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # 构造 prompt（基于统计信息）
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.horizons)} steps given the previous {str(self.input_size)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}"
                "Input images: "
                f"Input images: these are sky images of size 64 taken by a ground-based fish-eye camera at approximately 125 meters distance, capturing the power output of a 30 kW rooftop photovoltaic array. <|<end_prompt>|>"
            )
            prompt.append(prompt_)
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        prompt_llm = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings_llm = self.llm_model.get_input_embeddings()(prompt_llm.to(x_enc.device))
        # [num_soft, d_llm] -> [batch*, num_soft, d_llm]
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(prompt_embeddings_llm.size(0), -1, -1)
        # 拼接 soft prompt 在最前
        prompt_embeddings_llm = torch.cat([soft_prompt_batch, prompt_embeddings_llm], dim=1)
        x_enc = x_enc.to(torch.bfloat16)
        ts_out = self.patchtst(x_enc)
        aligned_ts = self.ts_projection(ts_out)

        # 图像特征提取与对齐
        imgs = imgs.to(torch.bfloat16)  # 将输入图像转换为 bfloat16
        B, T, C, H, W = imgs.shape  # 假设 imgs 原始形状为 [B, T, C, H, W]
        imgs_reshaped = imgs.view(B * T, C, H, W)
        image_forward_outs = self.vision_tower.vision_model(imgs_reshaped, output_hidden_states=True)
        img_feats_all = image_forward_outs.last_hidden_state[:, :, :]
        bs, ts_tokens, d_llm = img_feats_all.shape
        img_feats_all = img_feats_all.view(B, T, ts_tokens, d_llm)
        img_feats_all = img_feats_all.reshape(B, T * ts_tokens, d_llm)
        pos_emb = Rotate_PositionEmbedding(d_llm, max_len=img_feats_all.size(1)).to(img_feats_all.device)
        pos_emb = pos_emb.to(img_feats_all.dtype)
        img_feats_all = img_feats_all + pos_emb  # 广播 [B, L, D] + [1, L, D]
        # 卷积降低维度
        img_feats_all = img_feats_all.permute(0, 2, 1).contiguous()
        #　3x3 卷积
        img_feats_all = self.conv(img_feats_all)
        img_feats_all = img_feats_all.permute(0, 2, 1).contiguous()
        img_feats_all = self.connector(img_feats_all)
        input = torch.cat([prompt_embeddings_llm, img_feats_all], dim=1)
        
        if self.wio_img:
            input = prompt_embeddings_llm
        elif self.wio_prompt:
            input = img_feats_all
        fused_llm_out = self.llm_model(inputs_embeds=input).last_hidden_state
        fused_llm_out = self.llm_out_projection(fused_llm_out)
        fused_ts, _ = self.cma(aligned_ts, fused_llm_out, fused_llm_out)  # [B, L_ts, d_llm]
        final_ts = aligned_ts + fused_ts
        # final_ts = self.normalize(final_ts)
        if self.wio_ts:
            final_ts = fused_llm_out[:,:self.patch_nums,:]
        if self.only_ts:
            final_ts = aligned_ts
        dec_out = self.output_projection(final_ts)  # 此处根据你的 Head 定义调整
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def forward(self, imgs, x):
        return self.forecast(imgs, x)