import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Image encoder: Depthwise Separable CNN (解像度非依存) ---------
class DWConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SimpleDWEncoder(nn.Module):
    """
    入力: (B, 3, H, W) → 出力: (B, feat_dim)
    """
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        c = [32, 64, 128, 256]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWConv2d(c[0], c[1], s=2),   # ↓1/2
            DWConv2d(c[1], c[2], s=2),   # ↓1/2
            DWConv2d(c[2], c[3], s=2),   # ↓1/2
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c[-1], feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # (B, 256)
        return self.proj(x)          # (B, feat_dim)


# --------- View attention pooling: 時刻内で num_images を学習的に集約 ---------
class ViewAttentionPool(nn.Module):
    """
    x: (B, V, T, F) -> pooled: (B, T, F)
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, x):
        B, V, T, Fdim = x.shape
        scores = self.score(x.view(B*V*T, Fdim)).view(B, V, T, 1)  # (B,V,T,1)
        attn = F.softmax(scores, dim=1)                             # Vに沿って正規化
        pooled = (attn * x).sum(dim=1)                              # (B,T,F)
        return pooled


# --------- Temporal Conv Network (TCN): 1D 膨張Conv＋残差 ---------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=None, dropout=0.0):
        super().__init__()
        # same padding for dilation
        if p is None:
            p = (k - 1) * d // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=p, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=p, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):        # x: (B, C, T)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.act(self.bn2(self.conv2(y)))
        y = self.dropout(y)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(y + res) # (B, out_ch, T)


class TCN(nn.Module):
    def __init__(self, in_ch, channels=(256, 256, 256), k=3, base_dilation=1, dropout=0.0):
        super().__init__()
        layers = []
        c_in = in_ch
        d = base_dilation
        for c_out in channels:
            layers.append(TemporalBlock(c_in, c_out, k=k, d=d, dropout=dropout))
            c_in = c_out
            d *= 2  # 1,2,4,... で受容野を急速拡大
        self.net = nn.Sequential(*layers)
        self.out_ch = c_in

    def forward(self, x):  # (B, C_in, T)
        return self.net(x) # (B, C_out, T)


# --------- メイン: 高速・効果的な時系列モデル ---------
class CnnDwTcnPolicy(nn.Module):
    """
    CNN(Depthwise) + ViewAttentionPool + TCN + MLP head
    入出力:
      state_seq:  (B, n_obs_steps, state_dim)
      images_seq: (B, ?, ?, C, H, W)  ※ num_images × n_obs_steps
      return:     (B, n_action_steps, action_dim)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,      # 読み出しMLPの隠れ次元群
        state_feature_dim,
        img_feat_dim=256,     # 画像特徴次元（SimpleDWEncoderの出力）
        tcn_channels=(256, 256, 256),  # TCNの各層チャネル
        tcn_dropout=0.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_images = num_images

        # --- per-image encoder ---
        self.image_feature_extractor = SimpleDWEncoder(in_ch=3, feat_dim=img_feat_dim)
        self.view_pool = ViewAttentionPool(img_feat_dim)

        # --- per-step state embed ---
        self._has_state = state_dim > 0
        if self._has_state:
            self.state_step_embed = nn.Sequential(
                nn.Linear(state_dim, state_feature_dim),
                nn.ReLU(inplace=True),
            )
            per_step_dim = img_feat_dim + state_feature_dim
        else:
            self.state_step_embed = None
            per_step_dim = img_feat_dim

        # --- TCN over time ---
        self.tcn = TCN(in_ch=per_step_dim, channels=tcn_channels, k=3, base_dilation=1, dropout=tcn_dropout)
        tcn_out_ch = self.tcn.out_ch

        # --- readout head (sequence 全体をflatten→MLP) ---
        head_in = tcn_out_ch * n_obs_steps
        dims = [head_in] + list(hidden_dim_list) + [action_dim * n_action_steps]
        mlp = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                mlp.append(nn.ReLU(inplace=True))
        self.linear_layer_seq = nn.Sequential(*mlp)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        """
        state_seq:  (B, T, state_dim)
        images_seq:(B, d1, d2, C, H, W) with d1*d2 == num_images*n_obs_steps
        """
        B, d1, d2, C, H, W = images_seq.shape
        assert d1 * d2 == self.num_images * self.n_obs_steps, "num_images / n_obs_steps と images_seq の形が一致しません"
        V, T = self.num_images, self.n_obs_steps

        # (B, V, T, C, H, W)
        imgs = images_seq.view(B, V, T, C, H, W)

        # --- encode all images in one pass ---
        x = imgs.view(B * V * T, C, H, W)                       # (B*V*T, C, H, W)
        img_feat = self.image_feature_extractor(x)              # (B*V*T, F)
        img_feat = img_feat.view(B, V, T, -1)                   # (B, V, T, F)

        # --- view attention pooling (per time step) ---
        img_step_feat = self.view_pool(img_feat)                # (B, T, F)

        # --- state embedding (per time step) ---
        if self._has_state:
            st = self.state_step_embed(state_seq)               # (B, T, S)
            step_feat = torch.cat([img_step_feat, st], dim=-1)  # (B, T, F+S)
        else:
            step_feat = img_step_feat                           # (B, T, F)

        # --- TCN over time ---
        h = step_feat.transpose(1, 2)                           # (B, C_in, T)
        h = self.tcn(h)                                         # (B, C_out, T)
        h = h.flatten(1)                                        # (B, C_out*T)

        # --- readout ---
        out = self.linear_layer_seq(h)                          # (B, action_dim*n_action_steps)
        out = out.view(B, self.n_action_steps, -1)              # (B, n_action_steps, action_dim)
        return out
