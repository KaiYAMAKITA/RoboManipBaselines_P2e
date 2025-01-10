# Utilities

## Data processing
### Visualize npz file
```console
$ python ./visualize_data.py <npz_file>
```

### Renew old format data
```console
$ python renew_data.py <npz_file>
```

### Tile teleoperation videos
```console
$ python tile_teleop_videos.py <output_video_path> <npz_directory> --column_num 2 --envs env0 env1 env4 env5
```
The options `--column_num` and `--envs` can be omitted.

### Tile rollout videos
The input is a video consisting of a sequence of multiple rollouts, and the output is a tiled video of each rollout.
```console
$ python tile_rollout_videos.py <input_video_path> --output_file_name <output_video_path> --task_success_list 1 0 1 0 1 1 --column_num 3
```
The options `--output_file_name`, `--task_success_list`, and `--column_num` can be omitted.

By default, the video separation times are automatically determined by detecting restarts of windows in the video, but can be specified explicitly by adding the `--task_period_list` option as follows:
```console
--task_period_list 00:00.00-00:11.00 00:14.00-00:27.50 00:30.20-00:42.50 00:45.70-00:58.50 01:01.70-01:13.50 01:16.70-01:28.00
```
