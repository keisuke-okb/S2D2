import os
import PIL
from PIL import Image
import argparse
import os


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def convert(args):
    images = os.listdir(args.input)
    os.makedirs(args.output, exist_ok=True)

    for image in images:
        img = Image.open(os.path.join(args.input, image)).convert("RGB")
        im_new = expand2square(img, (255, 255, 255))
        im_new.save(os.path.join(args.output, image), quality=98)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./datasets/zunko")
    parser.add_argument("--output", type=str, default="./datasets/zunko_square")
    parser.add_argument("--size", type=int, default=512)

    args = parser.parse_args()
    convert(args)
