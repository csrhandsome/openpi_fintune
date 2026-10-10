> 本分支使用 `scripts/train.py` / `scripts/server.py` 和 `config/train/*.yaml`。运行本例前，先在该目录创建匹配机器人和 PyTorch checkpoint 的 YAML；旧的 Python 配置名称与 JAX 权重不可直接使用。参见根目录 README。

# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e src/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/server.py --config config/train/<robot>.yaml
```
