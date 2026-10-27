"""Link the project's data/force directory to an existing NAS mount."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nas-root", type=Path, default=Path("/mnt/nas"))
    parser.add_argument("--link", type=Path, default=Path(__file__).resolve().parents[2] / "data/force")
    args = parser.parse_args()
    target = args.nas_root.expanduser().resolve(strict=True)
    link = args.link.expanduser().absolute()
    if link.is_symlink() and link.resolve() == target:
        print(f"Already linked: {link} -> {target}")
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    print(f"Linked: {link} -> {target}")


if __name__ == "__main__":
    main()
