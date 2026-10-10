"""Serve from a server YAML: python scripts/server.py --config config/server/openpi.yaml."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tyro

from openpi.serving.runner import Args
from openpi.serving.runner import serve


def main(args: Args | None = None):
    serve(args if args is not None else tyro.cli(Args))


if __name__ == "__main__":
    main()
