from mini_jepa.images import create_demo_image, save_image


def main() -> None:
    image = create_demo_image(width=128, height=128)
    save_image(image, "outputs/demo_image.png")


if __name__ == "__main__":
    main()

