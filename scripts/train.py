import argparse
from pathlib import Path

from mini_jepa.images import create_demo_image, save_image
from mini_jepa.masks import generate_block_mask, save_mask_png
from mini_jepa.training import train_on_dir, train_on_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--image", default="outputs/demo_image.png")
    parser.add_argument("--mask", default="outputs/python_mask.png")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.image_dir is not None:
        losses = train_on_dir(
            image_dir=args.image_dir,
            patch_size=args.patch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    else:
        ensure_demo_inputs(args.image, args.mask)
        losses = train_on_image(
            image_path=args.image,
            mask_path=args.mask,
            patch_size=args.patch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

    for epoch, loss in enumerate(losses, start=1):
        print(f"epoch {epoch:03d} | loss {loss:.6f}")


def ensure_demo_inputs(image_path: str, mask_path: str) -> None:
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if not image_path.exists():
        save_image(create_demo_image(width=128, height=128), image_path)

    if not mask_path.exists():
        mask = generate_block_mask(width=128, height=128, block_size=24, blocks=6, seed=42)
        save_mask_png(mask, mask_path)


if __name__ == "__main__":
    main()

