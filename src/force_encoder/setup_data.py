"""Link data/force or data/soft_sim to an existing local or NAS dataset directory."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nas-root", type=Path, default=None)
    parser.add_argument("--soft-sim-root", type=Path, default=None)
    parser.add_argument("--link", type=Path, default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.soft_sim_root is not None:
        raw_target = args.soft_sim_root
        default_link = project_root / "data/soft_sim"
    else:
        raw_target = args.nas_root if args.nas_root is not None else Path("/mnt/nas")
        default_link = project_root / "data/force"
    target = raw_target.expanduser().resolve(strict=True)
    link = (args.link or default_link).expanduser().absolute()
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
