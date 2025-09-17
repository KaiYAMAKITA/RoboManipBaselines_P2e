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
    出力: (B, 512)  ※AdaptiveAvgPool2dで解像度非依存
    """
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 64, k=7, s=2, p=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, s=2),
            ConvBlock(128, 256, s=2),
            ConvBlock(256, 512, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)          # (B, 512, 1, 1)
        return x.view(x.size(0), -1)  # (B, 512)


class TrainableReservoir(nn.Module):
    """
    “RC風”のリザバー層（学習可能）。
    r_t = (1 - alpha) * r_{t-1} + alpha * tanh( x_t W_in^T + r_{t-1} W^T + b )
    """
    def __init__(self, input_dim, reservoir_dim, alpha=0.5, rho=0.9, power_iter=10):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.alpha = alpha

        self.W_in = nn.Parameter(torch.empty(reservoir_dim, input_dim))
        self.W = nn.Parameter(torch.empty(reservoir_dim, reservoir_dim))
        self.b = nn.Parameter(torch.zeros(reservoir_dim))

        nn.init.kaiming_uniform_(self.W_in, nonlinearity="tanh")
        # ランダム初期化 → スペクトルノルム近似でスケール（安定化）
        nn.init.normal_(self.W, mean=0.0, std=1.0 / reservoir_dim)
        with torch.no_grad():
            self._scale_spectral_norm(self.W, target=rho, iters=power_iter)

    @staticmethod
    def _scale_spectral_norm(W, target=0.9, iters=10):
        # パワーイテレーションで最大特異値を近似し、targetにスケーリング
        device = W.device
        N = W.shape[0]
        u = torch.randn(N, device=device)
        u = u / (u.norm() + 1e-12)
        for _ in range(iters):
            v = torch.mv(W.t(), u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(W, v)
            u = u / (u.norm() + 1e-12)
        sigma = torch.dot(u, torch.mv(W, v)).abs() + 1e-12  # 最大特異値の近似
        W.mul_(target / sigma)

    def forward(self, x_seq):
        """
        x_seq: (B, T, D_in)
        return: r_seq: (B, T, N)
        """
        B, T, D = x_seq.shape
        N = self.reservoir_dim
        r_prev = x_seq.new_zeros(B, N)
        r_list = []
        for t in range(T):
            x_t = x_seq[:, t, :]  # (B, D)
            pre = torch.addmm(self.b, r_prev, self.W.t()) + x_t @ self.W_in.t()
            r_t = (1.0 - self.alpha) * r_prev + self.alpha * torch.tanh(pre)
            r_list.append(r_t)
            r_prev = r_t
        r_seq = torch.stack(r_list, dim=1)  # (B, T, N)
        return r_seq


class CnnReservoirPolicy(nn.Module):
    """
    CNN + Trainable Reservoir + Linear Readout
    入力/出力形状は CnnmlpPolicy と同じ。
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,     # [reservoir_dim, head_hidden1, head_hidden2, ...]
        state_feature_dim,
        alpha=0.5,
        rho=0.9,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_images = num_images

        # --- state embedding (時刻ごと) ---
        self._has_state = state_dim > 0
        if self._has_state:
            self.state_step_embed = nn.Sequential(
                nn.Linear(state_dim, state_feature_dim),
                nn.ReLU(),
            )
            state_step_dim = state_feature_dim
        else:
            self.state_step_embed = None
            state_step_dim = 0

        # --- image encoder (各画像 → 512次元) ---
        self.image_feature_extractor = SimpleCNN(in_ch=3, out_dim=512)
        img_feat_dim = self.image_feature_extractor.out_dim

        # --- reservoir ---
        reservoir_dim = hidden_dim_list[0] if len(hidden_dim_list) > 0 else 1024
        self.reservoir = TrainableReservoir(
            input_dim=img_feat_dim + state_step_dim,
            reservoir_dim=reservoir_dim,
            alpha=alpha,
            rho=rho,
        )

        # --- readout head ---
        head_in = n_obs_steps * reservoir_dim
        head_hiddens = hidden_dim_list[1:] if len(hidden_dim_list) > 1 else []
        dims = [head_in] + list(head_hiddens) + [action_dim * n_action_steps]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.linear_layer_seq = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        """
        state_seq:  (B, n_obs_steps, state_dim)
        images_seq:(B, ?, ?, C, H, W)  → (B, num_images, n_obs_steps, C, H, W) を想定
        return:    (B, n_action_steps, action_dim)
        """
        B, d1, d2, C, H, W = images_seq.shape
        assert d1 * d2 == self.num_images * self.n_obs_steps, "num_images / n_obs_steps の指定と images_seq の形が一致しません"
        # 形を明示（時刻軸で集約しやすく）
        imgs = images_seq.reshape(B, self.num_images, self.n_obs_steps, C, H, W)

        # --- 画像特徴: 各画像 → CNN → (時刻ごとに平均) ---
        x = imgs.reshape(B * self.num_images * self.n_obs_steps, C, H, W)
        feat = self.image_feature_extractor(x)  # (B*num_images*n_obs, 512)
        feat = feat.view(B, self.num_images, self.n_obs_steps, -1)      # (B, num_images, T, 512)
        img_step_feat = feat.mean(dim=1)                                 # (B, T, 512)

        # --- 状態埋め込み（時刻ごと） ---
        if self._has_state:
            state_step = self.state_step_embed(state_seq)                # (B, T, state_feature_dim)
            step_inputs = torch.cat([img_step_feat, state_step], dim=-1) # (B, T, 512+state_dim)
        else:
            step_inputs = img_step_feat                                  # (B, T, 512)

        # --- Reservoir 展開 ---
        r_seq = self.reservoir(step_inputs)                              # (B, T, N)

        # --- 読み出し ---
        r_flat = r_seq.reshape(B, -1)                                    # (B, T*N)
        action_seq = self.linear_layer_seq(r_flat)                       # (B, action_dim*n_action_steps)
        action_seq = action_seq.view(B, self.n_action_steps, -1)         # (B, n_action_steps, action_dim)
        return action_seq
