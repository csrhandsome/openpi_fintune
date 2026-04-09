最终场景和效果：
机械臂按预设指令"place"抓取物体，沿直线向左篮子平稳移动。操作员来不及下达完整指令，本能短促发出一声"诶？"麦克风捕捉到语音特征 → 系统即时重定向 → 机械臂无停顿、不回程，半空中丝滑右拐画弧，绕开障碍，精准落入右篮。

数据采集：
对于声音，相关采集的脚本在/home/three/DataFromRoot/codes/grade_two/my_projects/data_collect/vr_openpi_lerobot_joint_audio.py

每个 episode 对应一个音频同步文件 `audio/episode_xxxxxx.sync.json`：
```json
{
  "episode_index": 123,
  "task": "place",
  "control_frequency": 20.0,
  "audio_path": "audio/episode_000123.wav",
  "audio_metadata_path": "audio/episode_000123.audio.json",
  "audio_start_monotonic_ns": 123456789000000,
  "audio_stop_monotonic_ns": 123456790000000,
  "episode_start_monotonic_ns": 123456789100000,
  "video_frames": 240,
  "frame_records": [
    {
      "frame_index": 0,
      "host_frame_monotonic_ns": 123456789123456,
      "external_camera_timestamp": 17123456.12,
      "wrist_camera_timestamp": 17123456.15,
      "external_host_capture_monotonic_ns": 123456789120000,
      "wrist_host_capture_monotonic_ns": 123456789121000,
      "joint_position": [q1, q2, q3, q4, q5, q6, q7],
      "gripper_position": 1.0
    }
  ],
  "camera_serials": {
    "external": "EXTERNAL_SERIAL",
    "wrist": "WRIST_SERIAL"
  }
}
```
每个 episode 的音频 metadata 文件 `audio/episode_xxxxxx.audio.json`：
```json
{
  "audio_path": ".../episode_000123.wav",
  "sample_rate": 16000,
  "channels": 1,
  "dtype": "int16",
  "audio_start_monotonic_ns": 123456789000000,
  "audio_stop_monotonic_ns": 123456790000000,
  "num_chunks": 160,
  "num_samples": 160000,
  "chunk_timestamps_ns": [123456789000100, 123456789100200],
  "input_buffer_adc_times": [null, null]
}
```

利用 `sync.json` 里的 `audio_start_monotonic_ns` 和 `frame_records[*].host_frame_monotonic_ns`，把整段 `episode.wav` 对齐到帧区间后再自动切成短 `.wav` 片段，并把这些短片段路径写进给 VLAS 看的训练配置文件里。


录制内容：
├── 视觉流（顶部相机，20hz）
├── 音频流（麦克风，连续录制）
└── 机械臂动作流（20hz）

你的操作：
├── 正常组：不发声，机械臂走左边
└── 干预组：喊"哎"，同时遥操作引导机械臂走右边

模型架构：
Whisper encoder提取音频embedding，加一个轻量projector接进π0
音频embedding → 小projector → 作为额外token注入π0的attention
不动π0主体权重，只训练projector

prefix
sound->whisper encoder->mlp->soundprefix
concat(soundprefix,prefix)->prefix_sound
if enable_moe:
    prefix_sound->moe

训练推理:
`src/openpi/models/pi0_audio.py`，扩展 `Observation`，只在 prefix 侧加入 audio token。将`src/openpi/models/limoe_simple.py`作为可选的模块在prefix+audiofix之后
数据预处理：
新增对应的 policy / data transform在`src/openpi/transforms.py`
在`src/openpi/shared/audio_tools.py`里对声音进行处理
1:把每个 sample 对齐
2:音频窗口变成一维的向量，窗口大小如果为1s,那么相当于训推的时候都录制1s之后将一维的向量传入进行处理
训推设置
新增 `Pi0AudioConfig` / train config，让训练和推理真正走 `pi0_audio.py`。
新增 `Pi0AudioWeightLoader`，加载 base checkpoint，同时保留新加的 audio projector 参数。
