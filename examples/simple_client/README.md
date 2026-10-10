> 本分支使用 `scripts/train.py` / `scripts/server.py` 和 `config/train/*.yaml`。运行本例前，先在该目录创建匹配机器人和 PyTorch checkpoint 的 YAML；旧的 Python 配置名称与 JAX 权重不可直接使用。参见根目录 README。

# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
uv run scripts/server.py --config config/train/<robot>.yaml
```
