"""
纯 Pytorch 手写 Vision Transformer （ViT）图片分类模型
"""

import torch 
import torch.nn as nn 
from modelscope import snapshot_download

class PatchEmbedding(nn.Module):
    """用卷积将图片切成互不重叠的 patch 并线性映射到 embed_dim """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size 必须能整除 patch_size")
        self.num_patches = (img_size // patch_size) ** 2 
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, num_patch_h, num_patch_w]
        return x.flatten(2).transpose(1, 2) # -> [B, embed_dim, num_patch_h * num_patch_w] -> [B, num_patch_h * num_patch_w, embed_dim]


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, drop_rate=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须整除 num_heads")
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads 
        self.scale = self.head_dim ** -0.5 
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        B, S, _ = x.shape 
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, self.num_heads, S, self.head_dim]
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-LN 结构: LN -> Attention / MLP -> 残差"""
    def __init__(self, embed_dim, num_heads, mlp_ratio, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x 


class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x): # x: [B, C, img_size, img_size]
        B = x.shape[0]
        x = self.patch_embed(x) # [B, S, D]
        cls_token = self.cls_token.expand(B, -1, -1) # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1) # [B, S+1, D]
        x = self.pos_drop(x + self.pos_embed) # [B, S+1, D]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) # [B, S+1, D]
        return self.head(x[:, 0])

# --- 构建模型&加载权重 --- 
def build_model(args, num_classes):
    return VisionTransformer(
        img_size=args.img_size, patch_size=args.patch_size,
        num_classes=args.num_classes, embed_dim=args.embed_dim,
        depth=args.depth, num_heads=args.num_heads, 
        mlp_ratio=args.mlp_ratio, drop_rate=args.drop_rate
    )

# def load_pretrained()


if __name__ == "__main__":

    # 测试 PatchEmbedding
    B, C, img_size, patch_size, embed_dim = 4, 3, 32, 4, 128
    patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                 in_channels=C, embed_dim=embed_dim)
    img = torch.randn(B, C, img_size, img_size)
    result = patch_embed(img)
    print(result.shape)

    # 测试 VisionTransformer
    model = VisionTransformer(img_size=img_size, 
                              patch_size=patch_size, 
                              embed_dim=embed_dim)
    img = torch.randn(B, C, img_size, img_size)
    logits = model(img)
    print(logits.shape)

    # 下载模型
    model_id = "iic/cv_vit-base_image-classification_Dailylife-labels"
    snapshot_download(model_id=model_id, cache_dir="/root/autodl-tmp/cache")

    # ViT-Base 固定配置
    vit_base_args = dict(
        
    )