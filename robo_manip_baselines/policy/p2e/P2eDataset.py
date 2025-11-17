import numpy as np
import torch
from attrdict import AttrDict

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    get_skipped_data_seq,
)

@torch.no_grad()
def build_p2e_attrdict_dataset(
    filenames,
    model_meta_info,
    enable_rmb_cache: bool = False,
    device: torch.device | str = "cpu",
):
    """
    戻り値: AttrDict({
        'state':  FloatTensor [N, L, state_dim],
        'action': FloatTensor [N, L, action_dim],
        'images': UInt8Tensor [N, L, N_cam, H, W, 3],
    })
    ※ 各エピソードの長さが異なる場合は、最後のフレームをリピートして L(=max長) に揃えます。
    """
    skip = model_meta_info["data"]["skip"]

    state_keys  = model_meta_info.get("state", {}).get("keys", [])
    action_keys = model_meta_info.get("action", {}).get("keys", [])
    cam_names   = model_meta_info.get("image", {}).get("camera_names", [])
    reward_keys = ["reward"]
    rwd_exist = False

    per_epi_state = []
    per_epi_action = []
    per_epi_images = []
    per_epi_reward = []
    lengths = []

    # 1) エピソードごとに [T_i, ...] を作る
    for fname in filenames:
        with RmbData(fname, enable_rmb_cache) as rmb:
            print(f"Loading episode from {fname}...")
            print("reards" if "reward" in rmb.keys() else "None")
            T = rmb[DataKey.TIME][::skip].shape[0]
            lengths.append(T)

            # --- state [T, Ds] ---
            if len(state_keys) == 0:
                state_np = np.zeros((T, 0), dtype=np.float32)
            else:
                state_np = np.concatenate(
                    [get_skipped_data_seq(rmb[key][:], key, skip)[:T] for key in state_keys],
                    axis=1,
                ).astype(np.float32)

            # --- action [T, Da] ---
            if len(action_keys) == 0:
                action_np = np.zeros((T, 0), dtype=np.float32)
            else:
                action_np = np.concatenate(
                    [get_skipped_data_seq(rmb[key][:], key, skip)[:T] for key in action_keys],
                    axis=1,
                ).astype(np.float32)

            # --- images [T, N_cam, H, W, 3] ---
            if len(cam_names) == 0:
                images_np = None
            else:
                cam_stacks = [
                    rmb[DataKey.get_rgb_image_key(cam)][::skip][:T]  # [T, H, W, 3] (uint8)
                    for cam in cam_names
                ]
                images_np = np.stack(cam_stacks, axis=0).transpose(1, 0, 2, 3, 4).astype(np.uint8)

            
            # --- reward [T, 1] ---
            if reward_keys[0] in rmb.keys():
              rwd_exist = True
              if len(reward_keys) == 0:
                  reward_np = None
              else:
                  reward_np = np.concatenate(
                      [get_skipped_data_seq(rmb[key][:], key, skip)[:T, None] for key in reward_keys],
                  ).astype(np.float32)
         
            per_epi_state.append(state_np)
            per_epi_action.append(action_np)
            per_epi_images.append(images_np)
            if reward_keys[0] in rmb.keys():
              per_epi_reward.append(reward_np)


    # 2) L を決定（最大長）
    L = max(lengths) if len(lengths) > 0 else 0
    N = len(filenames)

    # 3) 時間長を L にパディング（不足分は最後のフレームをリピート）
    def pad_to_L_edge(x, L):
        if x is None:
            return None
        T = x.shape[0]
        if T == L:
            return x
        if T == 0:
            # ありえないが安全策（edgeパディングできないため0埋め）
            out_shape = (L,) + x.shape[1:]
            return np.zeros(out_shape, dtype=x.dtype)
        pad_width = [(0, L - T)] + [(0, 0)] * (x.ndim - 1)
        # 'edge' で末尾フレームを繰り返す
        return np.pad(x, pad_width=pad_width, mode='edge')

    per_epi_state = [pad_to_L_edge(x, L) for x in per_epi_state]
    per_epi_action = [pad_to_L_edge(x, L) for x in per_epi_action]
    if rwd_exist:
      per_epi_reward = [pad_to_L_edge(x, L) for x in per_epi_reward]
    if any(img is not None for img in per_epi_images):
        # 画像が1つでもあるなら None のエピソードは空画像で用意（通常は起きない想定）
        first_img = next(img for img in per_epi_images if img is not None)
        N_cam, H, W = first_img.shape[1], first_img.shape[2], first_img.shape[3]
        per_epi_images = [
            pad_to_L_edge(
                (img if img is not None else np.zeros((0, N_cam, H, W, 3), dtype=np.uint8)),
                L
            )
            for img in per_epi_images
        ]
    else:
        per_epi_images = [None] * N

    # 4) [N, L, ...] にスタック
    # state/action
    if len(per_epi_state) > 0:
        state_dim = per_epi_state[0].shape[1]
        state = torch.from_numpy(np.stack(per_epi_state, axis=0)) if state_dim > 0 else torch.empty((N, L, 0), dtype=torch.float32)
    else:
        state = torch.empty((0, 0, 0), dtype=torch.float32)

    if len(per_epi_action) > 0:
        action_dim = per_epi_action[0].shape[1]
        action = torch.from_numpy(np.stack(per_epi_action, axis=0)) if action_dim > 0 else torch.empty((N, L, 0), dtype=torch.float32)
    else:
        action = torch.empty((0, 0, 0), dtype=torch.float32)

    # images
    if len(cam_names) == 0:
        images = torch.empty((N, L, 0, 0, 0, 0), dtype=torch.uint8)
    else:
        images = torch.from_numpy(np.stack(per_epi_images, axis=0))  # [N, L, N_cam, H, W, 3]
    
    # reward
    if rwd_exist:
      if len(per_epi_reward) > 0:
          reward_dim = per_epi_reward[0].shape[1]
          reward = torch.from_numpy(np.stack(per_epi_reward, axis=0)) if reward_dim > 0 else torch.empty((N, L, 0), dtype=torch.float32)
      else:
          reward = torch.empty((0, 0, 0), dtype=torch.float32)

    # 5) デバイス・dtype
    state  = state.to(dtype=torch.float32, device=device)
    action = action.to(dtype=torch.float32, device=device)
    images = images.to(device=device)  # 画像はuint8のまま
    if rwd_exist:
      reward = reward.to(dtype=torch.float32, device=device)

    print("states")
    print(state.shape)
    print("actions")
    print(action.shape)
    print("images")
    print(images.shape)
    if rwd_exist : #ここで上3つとデータ数で差が出ていたらrewardがきちんと記録されているかどうかを確認
      print("rewards")
      print(reward.shape)

    state = state[:, :L-1, :]
    action = action[:, :L-1, :]
    observation = images[:, :L-1, :, :, :, :]
    next_observation = images[:, 1:, :, :, :, :]
    if rwd_exist:
      reward = reward[:, :L-1, :]
    done = torch.zeros_like(reward, dtype=torch.float32)

    """
    return AttrDict(
        state=state,    # [N, L, Ds]
        action=action,  # [N, L, Da]
        images=images,  # [N, L, C, H, W, 3]
        reward=reward if rwd_exist else None,  # [N, L, 1]
    )"""
    if rwd_exist:
      return AttrDict(
        observation=observation,  # [N, L, C, H, W, 3]
        action=action,  # [N, L, Da]
        reward=reward,  # [N, L, 1]
        next_observation=next_observation,  # [N, L, C, H, W, 3]
        done=done,  # [N, L, 1]
      )
    
    return AttrDict(
        observation=observation,  # [N, L, C, H, W, 3]
        action=action,  # [N, L, Da]
        next_observation=next_observation,  # [N, L, C, H, W, 3]
        done=done,  # [N, L, 1]
    )

