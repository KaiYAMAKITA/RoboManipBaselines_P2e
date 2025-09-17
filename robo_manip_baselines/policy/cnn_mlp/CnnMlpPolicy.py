import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """
    入力: (B, C, H, W)
    出力: (B, D)  ※AdaptiveAvgPool2d(1)で解像度非依存
    """
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()
        # 出力次元は out_dim (= 最終チャネル数) に一致
        self.features = nn.Sequential(
            ConvBlock(in_ch,   64, k=7, s=2, p=3),   # 大きめの受容野 + ダウンサンプル
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64,     128, s=2),             # ↓1/2
            ConvBlock(128,    256, s=2),             # ↓1/2
            ConvBlock(256,    512, s=2),             # ↓1/2
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim  # 512

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)         # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 512)
        return x


class CnnMlpPolicy(nn.Module):
    """CNN + MLP policy. 入力/出力形状は元のMlpPolicyと同じ。"""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,
        state_feature_dim,
    ):
        super().__init__()

        # Setup Variable
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_images = num_images

        # State feature extractor（state_dim==0 の場合は None にして安全に）
        if state_dim > 0:
            self.state_feature_extractor = nn.Sequential(
                nn.Linear(state_dim * self.n_obs_steps, state_feature_dim),
                nn.ReLU(),
            )
            self._has_state = True
            state_feat_dim_for_concat = state_feature_dim
        else:
            self.state_feature_extractor = None
            self._has_state = False
            state_feat_dim_for_concat = 0

        # Image feature extractor（ResNetの代わりに軽量CNN）
        self.image_feature_extractor = SimpleCNN(in_ch=3, out_dim=512)
        image_feature_dim = self.image_feature_extractor.out_dim

        # Linear heads
        combined_feature_dim = state_feat_dim_for_concat + num_images * n_obs_steps * image_feature_dim
        linear_dim_list = [combined_feature_dim] + list(hidden_dim_list) + [action_dim * self.n_action_steps]

        layers = []
        for i in range(len(linear_dim_list) - 1):
            in_d, out_d = linear_dim_list[i], linear_dim_list[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(linear_dim_list) - 2:
                layers.append(nn.ReLU())
        self.linear_layer_seq = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        """
        state_seq:  (B, n_obs_steps, state_dim) を想定（元コード同様に flatten）
        images_seq: (B, ?, ?, C, H, W)  ※元コード同様、先頭2次元をflattenして処理
        return:     (B, n_action_steps, action_dim)
        """
        B, d1, d2, C, H, W = images_seq.shape  # 元コードと同じ受け方
        seq_len = d1 * d2                      # = num_images * n_obs_steps を想定

        # --- State feature ---
        if self._has_state:
            state_flat = state_seq.reshape(B, -1)                       # (B, n_obs_steps*state_dim)
            state_feature = self.state_feature_extractor(state_flat)    # (B, state_feature_dim)

        # --- Image feature（ベクトル化して一括処理） ---
        img_flat = images_seq.reshape(B * seq_len, C, H, W)            # (B*L, C, H, W)
        img_feat = self.image_feature_extractor(img_flat)               # (B*L, 512)
        img_feat = img_feat.view(B, seq_len, -1)                        # (B, L, 512)
        image_features = img_feat.reshape(B, -1)                        # (B, L*512) = (B, num_images*n_obs_steps*512)

        # --- Concat & head ---
        if self._has_state:
            combined_feature = torch.cat([state_feature, image_features], dim=1)
        else:
            combined_feature = image_features  # FIXME: state_dim==0 の場合はこちらのみ

        action_seq = self.linear_layer_seq(combined_feature)            # (B, action_dim * n_action_steps)
        action_seq = action_seq.reshape(B, self.n_action_steps, -1)     # (B, n_action_steps, action_dim)
        
        return action_seq
