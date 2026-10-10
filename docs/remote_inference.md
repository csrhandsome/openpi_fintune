
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

The server reads `config/server/openpi.yaml`, references its training YAML, and selects its latest PyTorch checkpoint.
Panda requests use `observation.exterior_image`, `observation.wrist_image_left`,
`observation.ee_pose`, `observation.gripper_position`, and `prompt`.
The YAML's action layout determines physical action dimensions. EE-delta models
return absolute EE/gripper actions. See [Panda setup](franka_ros_30hz.md).

```bash
uv run scripts/server.py --config config/server/openpi.yaml --port 8001
uv run scripts/server.py --config config/server/openpi.yaml --checkpoint /path/to/checkpoint
uv run scripts/server.py --config config/train/openpi_rlt.yaml --port 8001
```

The RLT YAML supplies `rlt.vla_checkpoint` and the readout directory. This mode
returns `z_rlt` and `ref_chunk` from the current observation, alongside `actions`.
All checkpoints must contain PyTorch `model.safetensors`; training configurations
come from YAML files under the repository-root `config/train/` directory.

For online actor training or frozen actor evaluation, use `config/server/openpi_rlt.yaml`.
This server requires episode lifecycle and executed-action feedback through `RLTClientPolicy`;
see [RLT online serving](rlt_online.md). A standalone readout server using the training YAML
continues to execute VLA actions.

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/src/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).
