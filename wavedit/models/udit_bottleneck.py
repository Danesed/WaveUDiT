import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _space_to_channel_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """(B,C,D,H,W) -> (B, C*r^3, D/r, H/r, W/r)."""
    B, C, D, H, W = x.shape
    x = x.view(B, C, D // r, r, H // r, r, W // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()      # B,C,rz,ry,rx, d,h,w
    return x.view(B, C * r * r * r, D // r, H // r, W // r)


def _channel_to_space_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """Inverse of _space_to_channel_3d: (B, C*r^3, d,h,w) -> (B, C, d*r, h*r, w*r)."""
    B, Cr, d, h, w = x.shape
    C = Cr // (r * r * r)
    x = x.view(B, C, r, r, r, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()      # B,C, d,rz, h,ry, w,rx
    return x.view(B, C, d * r, h * r, w * r)


class RoPE3D(nn.Module):

    def __init__(self, d_head: int, base: float = 10000.0):
        super().__init__()
        self.rot = (d_head // 6) * 6                          # rotary dims (mult of 6)
        self.per = self.rot // 3                              # per-axis (even)
        if self.per > 0:
            inv = 1.0 / (base ** (torch.arange(0, self.per, 2).float() / self.per))
            self.register_buffer("inv_freq", inv, persistent=False)   # (per/2,)

    def _cos_sin(self, coords: torch.Tensor):
        # coords: (N,3) integer positions. Returns cos,sin of shape (N, rot).
        cs, sn = [], []
        inv = self.inv_freq.to(coords.device)
        for axis in range(3):
            a = coords[:, axis].float()[:, None] * inv[None, :]       # (N, per/2)
            cs.append(torch.cat([a.cos(), a.cos()], dim=-1))          # (N, per)
            sn.append(torch.cat([a.sin(), a.sin()], dim=-1))
        return torch.cat(cs, dim=-1), torch.cat(sn, dim=-1)

    def _rotate(self, x, cos, sin):
        xr, xp = x[..., :self.rot], x[..., self.rot:]
        chunks = xr.split(self.per, dim=-1)                  # one per axis
        rot_half = []
        for ch in chunks:
            x1, x2 = ch.chunk(2, dim=-1)
            rot_half.append(torch.cat([-x2, x1], dim=-1))
        xr = xr * cos + torch.cat(rot_half, dim=-1) * sin
        return torch.cat([xr, xp], dim=-1)

    def forward(self, q, k, coords):
        # q,k: (B, nh, N, d_head); coords: (N,3)
        if self.rot == 0:
            return q, k
        cos, sin = self._cos_sin(coords)
        cos = cos[None, None].to(q.dtype)
        sin = sin[None, None].to(q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class _MultiScaleDWFFN3D(nn.Module):
    def __init__(self, dim, mult=2, kernels=(5, 3, 1), dropout=0.0):
        super().__init__()
        hidden = dim * mult
        self.project_in = nn.Conv3d(dim, hidden, 1, bias=True)
        self.dwconv = nn.ModuleList([
            nn.Conv3d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=True)
            for k in kernels])
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.project_out = nn.Conv3d(hidden, dim, 1, bias=True)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, x):
        x = self.project_in(x)
        x = sum(dw(x) for dw in self.dwconv)
        x = self.drop(self.act(x))
        return self.project_out(x)


class UDiTBottleneck3D(nn.Module):

    def __init__(self, channels: int, downsample: int = 2, d_head: int = 64,
                 dropout: float = 0.0, ff_mult: int = 2, down_shortcut: bool = True,
                 ff_kernels=(5, 3, 1), mask_attn: bool = False, depth_attn: bool = False,
                 contra_attn: bool = False, tokenrep: bool = False, norm_groups: int = 8):
        super().__init__()
        if channels % d_head != 0:
            raise ValueError(f"channels ({channels}) must be divisible by d_head ({d_head}).")
        self.r = downsample
        self.d_head = d_head
        self.n_heads = channels // d_head
        self.dropout_p = dropout
        self.down_shortcut = down_shortcut
        self.mask_attn = mask_attn
        self.depth_attn = depth_attn
        self.contra_attn = contra_attn
        if contra_attn:
            self.contra_lambda = nn.Parameter(torch.zeros(self.n_heads))  # (nh,), init 0
            self.contra_log_sigma = nn.Parameter(torch.tensor(0.6931))    # sigma = exp(.) ~ 2 tokens
        if depth_attn:
            self.depth_mlp = nn.Sequential(
                nn.Linear(1, 16), nn.GELU(), nn.Linear(16, self.n_heads))
            nn.init.zeros_(self.depth_mlp[-1].weight)
            nn.init.zeros_(self.depth_mlp[-1].bias)
        r3 = downsample ** 3

        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.down = nn.Conv3d(channels * r3, channels, 1, bias=False)     # space->channel merge
        self.tokenrep = tokenrep
        if tokenrep:
            k = 2 * downsample - 1
            self.down_overlap = nn.Sequential(
                nn.Conv3d(channels, channels, k, stride=downsample, padding=downsample - 1,
                          groups=channels, bias=False),
                nn.Conv3d(channels, channels, 1, bias=False))
            nn.init.zeros_(self.down_overlap[-1].weight)
            self.up_overlap = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv3d(channels, channels * r3, 1, bias=False))
            nn.init.zeros_(self.up_overlap[-1].weight)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        # per-head learnable temperature for cosine-similarity attention (CLIP-style, clamped)
        self.logit_scale = nn.Parameter(torch.full([self.n_heads, 1, 1], math.log(10.0)))
        self.rope = RoPE3D(d_head)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.out_proj.weight)                              # -> identity at init
        self.up = nn.Conv3d(channels, channels * r3, 1, bias=False)       # channel->space split

        self.norm2 = nn.GroupNorm(norm_groups, channels)
        # U-DiT 'rep' multi-scale depthwise-conv FFN (zero-init out -> identity at init).
        self.ff = _MultiScaleDWFFN3D(channels, mult=ff_mult, kernels=ff_kernels, dropout=dropout)

        self._coord_cache = {}

    def _coords(self, d, h, w, device):
        key = (d, h, w)
        if key not in self._coord_cache:
            zz, yy, xx = torch.meshgrid(
                torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij")
            self._coord_cache[key] = torch.stack(
                [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1)
        return self._coord_cache[key].to(device)

    def _attn_bias(self, mask, depth, contra_center, contra_valid, B, d_, h_, w_, dtype, device):
        N = d_ * h_ * w_
        bias = None
        if self.mask_attn and mask is not None:
            mk = F.avg_pool3d(mask.to(dtype), self.r).flatten(1)          # (B,N) void fraction
            void = mk > 0.5                                              # True = void key
            keep_any = (~void).any(dim=1, keepdim=True)
            void = void & keep_any
            bias = torch.zeros(B, 1, 1, N, dtype=dtype, device=device)
            bias = bias.masked_fill(void.view(B, 1, 1, N), -1e9)
        if self.depth_attn and depth is not None:
            dp = F.avg_pool3d(depth.to(dtype), self.r).flatten(2).transpose(1, 2)  # (B,N,1)
            db = self.depth_mlp(dp).permute(0, 2, 1).unsqueeze(2)         # (B,nh,1,N)
            bias = db if bias is None else bias + db
        if self.contra_attn:
            c = self._coords(d_, h_, w_, device).float()                 # (N,3) z,y,x
            if contra_center is not None:                                # (B,) normalised [0,1] H
                center = (contra_center.to(device).float().clamp(0.0, 1.0) * (h_ - 1)).view(B, 1)
            else:                                                        # fall back to grid center
                center = torch.full((B, 1), (h_ - 1) / 2.0, device=device)
            qpos = c.unsqueeze(0).expand(B, -1, -1).clone()             # (B,N,3)
            qpos[:, :, 1] = 2.0 * center - c[:, 1].view(1, N)           # mirror query H about midline
            ck = c.unsqueeze(0).expand(B, -1, -1)                       # (B,N,3) keys
            # direct squared distance (avoids torch.cdist's 0/0 backward at coincident points)
            d2 = (qpos.unsqueeze(2) - ck.unsqueeze(1)).pow(2).sum(-1)   # (B,N,N)
            sigma = self.contra_log_sigma.exp().clamp(min=0.5)
            g = torch.exp(-d2 / (2.0 * sigma * sigma)).to(dtype)        # (B,N,N), peak 1 at mirror
            # keep the assembled bias in the query dtype so CUDA does not drop off the fused kernel
            cbias = self.contra_lambda.view(1, self.n_heads, 1, 1).to(dtype) * g.unsqueeze(1)  # (B,nh,N,N)
            if contra_valid is not None:                                # key-side symmetry confidence
                valid = F.avg_pool3d(contra_valid.to(dtype), self.r).flatten(1).view(B, 1, 1, N)
                cbias = cbias * valid                                   # suppress where mirror invalid
            bias = cbias if bias is None else bias + cbias              # -> (B,nh,N,N)
        return bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                depth: torch.Tensor = None, contra_center: torch.Tensor = None,
                contra_valid: torch.Tensor = None) -> torch.Tensor:
        B, C, D, H, W = x.shape
        r = self.r
        if D % r or H % r or W % r:
            raise ValueError(
                f"UDiTBottleneck3D needs bottleneck dims divisible by {r}, got {(D, H, W)}.")

        # --- downsampled global attention (the U-DiT move) ---
        xn = self.norm1(x)
        h = self.down(_space_to_channel_3d(xn, r))                        # (B,C,d,h,w)
        if self.down_shortcut:                                            # U-DiT down_shortcut=1
            h = h + F.avg_pool3d(xn, r)
        if self.tokenrep:                                                 # zero-init -> 0 at init
            h = h + self.down_overlap(xn)
        d_, h_, w_ = D // r, H // r, W // r
        coords = self._coords(d_, h_, w_, x.device)                       # (N,3)
        tok = h.flatten(2).transpose(1, 2)                                # (B,N,C)
        qkv = self.qkv(tok).reshape(B, -1, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # (B,nh,N,d_head)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q, k = self.rope(q, k, coords)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp().view(1, self.n_heads, 1, 1)
        q = q * scale
        attn_bias = self._attn_bias(mask, depth, contra_center, contra_valid,
                                    B, d_, h_, w_, q.dtype, x.device)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=1.0,
            dropout_p=self.dropout_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, -1, C)                           # (B,N,C)
        o = self.out_proj(o)                                              # zero-init
        o = o.transpose(1, 2).reshape(B, C, d_, h_, w_)
        up = self.up(o)                                                   # (B,C*r^3,d,h,w)
        if self.tokenrep:                                                 # zero-init -> 0 at init
            up = up + self.up_overlap(o)
        o = _channel_to_space_3d(up, r)                                   # (B,C,D,H,W)
        x = x + o

        # --- feed-forward (full bottleneck resolution) ---
        x = x + self.ff(self.norm2(x))
        return x
