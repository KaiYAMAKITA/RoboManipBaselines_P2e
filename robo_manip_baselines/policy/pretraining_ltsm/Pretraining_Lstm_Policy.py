import torch
import torch.nn as nn


class Pretraining_Lstm_Policy(nn.Module):
    """LSTM-based policy using only state sequence."""

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
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # ---- state の特徴抽出部を LSTM に変更 ----
        # 入力: (batch, seq_len=n_obs_steps, state_dim)
        # 出力: 最終ステップの hidden state -> (batch, state_feature_dim)
        self.state_lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=state_feature_dim,
            batch_first=True,
        )

        # ---- 画像は今回使わないので feature extractor は無し ----
        # num_images / 画像は forward の引数だけ残して無視します。

        # ---- 全結合層（出力は action_seq をまとめたベクトル） ----
        # 以前は [state_feature_dim + image_feature_dim] だったが
        # 今回は state_feature_dim のみを入力とする。
        linear_dim_list = (
            [state_feature_dim]
            + hidden_dim_list
            + [action_dim * self.n_action_steps]
        )

        linear_layers = []
        for linear_idx in range(len(linear_dim_list) - 1):
            input_dim = linear_dim_list[linear_idx]
            output_dim = linear_dim_list[linear_idx + 1]
            linear_layers.append(nn.Linear(input_dim, output_dim))
            if linear_idx < len(linear_dim_list) - 2:
                linear_layers.append(nn.ReLU())

        self.linear_layer_seq = nn.Sequential(*linear_layers)

        # Initialize weights (Linear だけ独自初期化)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        """
        state_seq:
            - これまで通り 2D / 3D どちらでも受けられるようにしておきます。
              (B, n_obs_steps * state_dim) も (B, n_obs_steps, state_dim) もOK
        images_seq:
            - 形式は維持するが、今回は使わない。
        戻り値:
            action_seq: (batch_size, n_action_steps, action_dim)
        """
        # ---- state の形整形 ----
        batch_size = state_seq.shape[0]

        if state_seq.dim() == 2:
            # 旧 MLP 実装と同じく、flatten 済み (B, n_obs_steps * state_dim) にも対応
            state_seq = state_seq.view(batch_size, self.n_obs_steps, self.state_dim)
        elif state_seq.dim() == 3:
            # (B, T, D) 想定。T と n_obs_steps が違うならそのまま T を使って LSTM にかける。
            # （必要なら assert T == self.n_obs_steps を入れてもよい）
            pass
        else:
            raise ValueError(
                f"state_seq must be 2D or 3D tensor, but got shape {state_seq.shape}"
            )

        # ---- LSTM に通して最後の hidden を state feature として使う ----
        # state_seq: (B, T, state_dim)
        _, (h_n, _) = self.state_lstm(state_seq)
        # h_n: (num_layers * num_directions, B, state_feature_dim)
        state_feature = h_n[-1]  # (B, state_feature_dim)

        # ---- 画像は無視（フォーマット維持のため引数だけ受け取る） ----
        _ = images_seq  # unused

        # ---- MLP でアクション列を一括生成 ----
        # ここは元の実装と同じで、
        #   state_feature -> (action_dim * n_action_steps)
        # に写像してから reshape
        action_flat = self.linear_layer_seq(state_feature)
        action_seq = action_flat.view(batch_size, self.n_action_steps, -1)

        return action_seq
