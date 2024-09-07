import torch
import torch.nn as nn


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        
        img_size = _make_tuple(img_size)
        patch_size = _make_tuple(patch_size)

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)
    

# VitMLP
class VitMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))


# Vit Block
class VitBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = VitMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, x, valid_lens=None):
        attn_output, _ = self.attention(*([self.ln1(x)] * 3), need_weights=False)
        x = x + attn_output
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout, use_bias=False, num_classes=10):
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))

        num_steps = self.patch_embedding.num_patches + 1
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"vitblk{i}", VitBlock(
                num_hiddens,
                num_hiddens,
                mlp_num_hiddens,
                num_heads,
                blk_dropout,
                use_bias
            ))
        
        self.head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        # append the cls token to patches
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1)
        x = self.dropout(x + self.pos_embedding)
        for blk in self.blks:
            x = blk(x)
        
        return self.head(x[:, 0])


if __name__ == "__main__":
    img_size = 32
    patch_size = 1
    num_hiddens = 192
    batch_size = 1024
    mlp_num_hiddens = 768
    num_heads = 8
    num_blks = 12
    emb_dropout = 0.1
    blk_dropout = 0.1
    lr = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.zeros(batch_size, 3, img_size, img_size)

    # patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
    # output = patch_emb(X)
    # # output => (4, 196, 512)
    # print(output.shape)

    # # vision transformer encoder blocks don't change input shape
    # encoder_blk = VitBlock(512, 512, 48, num_heads=8, dropout=0.5)
    # encoder_blk.to(device)
    # output = output.to(device)
    # output2 = encoder_blk(output)
    # print(output2.shape)
    # assert output.shape == output2.shape

    model = ViT(
        img_size,
        patch_size,
        num_hiddens,
        mlp_num_hiddens,
        num_heads,
        num_blks,
        emb_dropout,
        blk_dropout
    )

    model.to(device)

    # print(model)
    res = model(X.to(device))
    print(res.shape)