import argparse
import subprocess
from pathlib import Path

from mini_jepa.images import create_demo_image, save_image
from mini_jepa.training import train_on_image
from mini_jepa.visualization import save_visualization
from mini_jepa.images import load_rgb_image
from mini_jepa.masks import load_mask_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "outputs/demo_image.png"
    mask_path = repo_root / "outputs/rust_mask.png"
    visualization_path = repo_root / "outputs/rust_masked_demo.png"

    save_image(create_demo_image(args.width, args.height), image_path)
    run_rust_mask_cli(repo_root, args)

    image = load_rgb_image(image_path)
    mask = load_mask_png(mask_path)
    save_visualization(image, mask, visualization_path)

    losses = train_on_image(
        image_path=image_path,
        mask_path=mask_path,
        epochs=args.epochs,
    )

    for epoch, loss in enumerate(losses, start=1):
        print(f"epoch {epoch:03d} | loss {loss:.6f}")

    print(f"demo image: {image_path}")
    print(f"rust mask: {mask_path}")
    print(f"visualization: {visualization_path}")


def run_rust_mask_cli(repo_root: Path, args: argparse.Namespace) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "--manifest-path",
            "rust/mask-cli/Cargo.toml",
            "--",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--block-size",
            str(args.block_size),
            "--blocks",
            str(args.blocks),
            "--seed",
            str(args.seed),
            "--output",
            "rust_mask.png",
        ],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
