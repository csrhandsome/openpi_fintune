> 本分支使用 `scripts/train.py` / `scripts/server.py` 和 `config/train/*.yaml`。运行本例前，先在该目录创建匹配机器人和 PyTorch checkpoint 的 YAML；旧的 Python 配置名称与 JAX 权重不可直接使用。参见根目录 README。

# Fine-Tuning on Custom DROID Datasets

Here we describe how to fine-tune a model on a custom (smaller) dataset collected on the DROID platform. Like for other datasets, we will first convert the custom DROID dataset to LeRobot and then fine-tune a model (pi05-droid) on it.

This branch reads LeRobot v3 datasets for both training and normalization statistics. Prepare the data in that format before training.


## Step 1: Converting your custom DROID dataset to LeRobot

We will use a small subset of the real DROID dataset for this example. This is a subset of just 30 demonstrations -- we assume that you will use your own dataset instead, but here is the command to download our subset (1.6GB):
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>
```

We will also download the language annotations for the DROID dataset so we can pair our demonstrations with language instructions. Again, for your own data you can manually enter your language instructions and don't need to download our annotations. To download the DROID language annotations (12MB), run:
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>
```

For your own dataset, make sure that each episode's directory contains a folder called `recordings/MP4` -- if not, you need to first run the MP4 video extraction (from SVO files) using the script [here](https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py).

Now, we will use the `convert_droid_data_to_lerobot.py` script to create a LeRobot version of this dataset (takes <5min for the 30 demonstrations):
```
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>
```

## Step 2: Run fine-tuning with your custom dataset

Create a training YAML under `config/train/` with `data.type: droid` and `data.repo_id` set to the converted dataset repository ID. Set the model, PyTorch checkpoint path, assets directory, and experiment name in that YAML; see the root README for the training schema. The DROID converter stores joint velocity actions and the DROID data transforms preserve those actions.

Compute normalization statistics, then launch training:
```
uv run scripts/compute_norm_stats.py --config config/train/<your_droid_config>.yaml
uv run scripts/train.py --config config/train/<your_droid_config>.yaml
```

Once trained, you can follow the instructions in [DROID inference instructions](README.md) to serve the policy and run it on the robot.
