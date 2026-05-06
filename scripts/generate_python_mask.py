import argparse

from mini_jepa.masks import generate_block_mask, save_mask_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/python_mask.png")
    args = parser.parse_args()

    mask = generate_block_mask(
        width=args.width,
        height=args.height,
        block_size=args.block_size,
        blocks=args.blocks,
        seed=args.seed,
    )
    save_mask_png(mask, args.output)


if __name__ == "__main__":
    main()

