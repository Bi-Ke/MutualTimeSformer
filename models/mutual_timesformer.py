"""
    Function Mutual TimeSformer module.
    
    Reference: https://github.com/lucidrains/TimeSformer-pytorch

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 16, 2021
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


# attention
def attn(q, k, v):
    # for cls_token: q:[16, 1, 64], k:[16, 801, 64], v:[16, 801, 64]
    sim = einsum('b i d, b j d -> b i j', q, k)  # [16, 1, 801]
    attn = sim.softmax(dim=-1)  # [16, 1, 801]
    out = einsum('b i j, b j d -> b i d', attn, v)  # [16, 1, 64]
    return out


class TimeAttention(nn.Module):
    def __init__(
            self,
            dim,  # 512.
            dim_head=64,
            heads=8,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads  # 64 x 8 = 512, Lastly, 512 -> dim.

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        """
        :param x: [2, 801, 512]
        :param einops_from: 'b (f n) d', b:batch_size, f: frames, n: the number of patches, d: dims.
        :param einops_to: '(b n) f d'
        :param einops_dims: n=n, the number of patches
        :return:
        """
        h = self.heads
        # [2, 801, 512] -> [2, 801, 512x3] -> [[2, 801, 512], [2, 801, 512], [2, 801, 512]]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # [16, 801, 64]

        q *= self.scale

        # split out classification token at index 0. cls_k: [16, 1, 64], q_: [16, 800, 64]
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)  # [16, 1, 64]

        # rearrange across time. 'b (f n) d' -> '(b n) f d'  -> [1600, 8, 64]
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]  # 1600 / 16 = 100, num_patches.
        # [16, 1, 64] -> [1600, 1, 64], extending cls to each time.
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)  # [1600, 9, 64]
        v_ = torch.cat((cls_v, v_), dim=1)  # [1600, 9, 64]

        # attention
        out = attn(q_, k_, v_)  # q_:[1600, 8, 64], k_:[1600, 9, 64], v_:[1600, 9, 64], -> [1600, 8, 64]

        # merge back time. time: '(b n) f d' -> 'b (f n) d'
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)  # [16, 800, 64]

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)  # [16, 801, 64]

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  # [2x8, 801, 64] -> [2, 801, 8x64] -> [2, 801, 512]

        # combine heads out
        return self.to_out(out)


class SpaceAttention(nn.Module):
    def __init__(
            self,
            dim,  # 512.
            dim_head=64,
            heads=8,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads  # 64 x 8 = 512, Lastly, 512 -> dim.

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        """
        :param x: [2, 801, 512]
        :param einops_from: 'b (f n) d', b:batch_size, f: frames, n: the number of patches, d: dims.
        :param einops_to: '(b f) n d'
        :param einops_dims: n=n: the number of patches
        :param n_h: the number of patches along image height.
        :param n_w: the number of patches along image width.
        :return: [2, 801, dim]
        """
        h = self.heads
        # [2, 801, 512] -> [2, 801, 512x3] -> [[2, 801, 512], [2, 801, 512], [2, 801, 512]]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # [16, 801, 64]

        q *= self.scale

        # split out classification token at index 0. cls_k: [16, 1, 64], q_: [16, 800, 64]
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)  # [16, 1, 64]

        # rearrange across space. space: 'b (f n) d' -> '(b f) n d', -> [128, 100, 64]
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across space and concat
        r = q_.shape[0] // cls_k.shape[0]  # 128 / 16 = 8 frames.
        # [16, 1, 64] -> [128, 1, 64]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        # q1_, k1_, v1_ for id1; q2_, k2_, v2_ for id2.
        q1_, q2_ = torch.chunk(q_, chunks=2, dim=1)  # [128, 50, 64]
        k1_, k2_ = torch.chunk(k_, chunks=2, dim=1)  # [128, 50, 64]
        v1_, v2_ = torch.chunk(v_, chunks=2, dim=1)  # [128, 50, 64]

        # for id1
        k22_ = torch.cat((cls_k, k2_), dim=1)  # [128, 51, 64]
        v22_ = torch.cat((cls_v, v2_), dim=1)  # [128, 51, 64]
        out1 = attn(q1_, k22_, v22_)  # [128, 50, 64]

        # for id2
        k11_ = torch.cat((cls_k, k1_), dim=1)  # [128, 51, 64]
        v11_ = torch.cat((cls_v, v1_), dim=1)  # [128, 51, 64]
        out2 = attn(q2_, k11_, v11_)  # [128, 50, 64]

        # attention
        out = torch.cat((out1, out2), dim=1)  # [128, 100, 64]

        # merge back space. space: '(b f) n d' -> 'b (f n) d'
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)  # [16, 800, 64]

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)  # [16, 801, 64]

        # merge back the heads. [2x8, 801, 64 -> [2, 801, 8x64] -> [2, 801, 512]
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  #

        # combine heads out
        return self.to_out(out)


# main classes
class MutualTimeSformer(nn.Module):
    def __init__(
            self,
            *,
            dim,  # used to
            num_frames,  # the number of frames extracted in a video.
            num_classes,  # the number of classed predicted.
            image_size=(480, 640),  # (height, width)
            patch_size=(16, 16),  # (height, width)
            channels=3,
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] / 2 % patch_size[1] == 0, '(Two people) Image dimensions must be divisible by the patch size.'

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size[0] * patch_size[1]  # using a vector to represent a patch.

        self.patch_size = patch_size  # (16, 16)
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TimeAttention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, SpaceAttention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        # b: batchsize, f: frames, _:channels, h: height, w:width, device: GPU, p1: patch1, p2: patch2
        b, f, _, h, w, *_, device, p1, p2 = *video.shape, video.device, *self.patch_size
        assert h % p1 == 0 and w / 2 % p2 == 0, \
            f'height {h} and width {w} of video must be divisible by the patch size {p1, p2}'

        n = (h // p1) * (w // p2)  # the number of patches.

        video1, video2 = torch.chunk(video, chunks=2, dim=-1)  # batch_size, frames, channels, height, width
        video1 = rearrange(video1, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p1, p2=p2)  # [2, 400, 9216]
        video2 = rearrange(video2, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p1, p2=p2)  # [2, 400, 9216]
        video = torch.cat([video1, video2], dim=1).contiguous()  # [2, 800, 9216]

        tokens = self.to_patch_embedding(video)  # [2, 800, 512]

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)  # [2, 1, 512]

        x = torch.cat((cls_token, tokens), dim=1)  # [2, 801, 512]
        x += self.pos_emb(torch.arange(x.shape[1], device=device))  # [2, 801, 512]

        # The following code will be revised.
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x  # [2, 801, 512]
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)


if __name__ == "__main__":
    from config import parser
    args = vars(parser.parse_args())
    device = torch.device('cuda:'+args['gpu_number'] if torch.cuda.is_available() else 'cpu')

    # note that id1 and id2 have the same size.
    # id1: [2, 8, 3, 480, 320], id2: [2, 8, 3, 480, 320]
    # video = torch.randn(2, 8, 3, 480, 640).to(device)  # [batch, frames, channels, height, width]
    video = torch.randn(6, 8, 3, 600, 400).to(device)

    model = MutualTimeSformer(
        dim=args['dim'],  # 512,
        image_size=(600, 400),  # args['img_size'],
        patch_size=(50, 50),  # args['patch_size'],
        num_frames=args['num_frame'],  # 8,
        num_classes=args['num_classes'],  # 10,
        depth=args['depth'],  # 10,  # the number of multiple head attention layers.
        heads=args['heads'],  # 7,  # Multiple Head Attention.
        dim_head=args['dim_head'],  # 64,
        attn_dropout=args['attn_dropout'],  # 0.1,
        ff_dropout=args['ff_dropout'],  # 0.1
    )

    model = model.to(device)
    pred = model(video)  # [2, 10]
    print(pred.shape)
    print(pred)
