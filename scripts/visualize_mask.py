import argparse

from mini_jepa.images import load_rgb_image
from mini_jepa.masks import load_mask_png
from mini_jepa.visualization import save_visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="outputs/demo_image.png")
    parser.add_argument("--mask", default="outputs/python_mask.png")
    parser.add_argument("--output", default="outputs/masked_demo.png")
    args = parser.parse_args()

    image = load_rgb_image(args.image)
    mask = load_mask_png(args.mask)
    save_visualization(image, mask, args.output)


if __name__ == "__main__":
    main()

