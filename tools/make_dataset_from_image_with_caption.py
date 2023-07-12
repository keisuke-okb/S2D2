import os
import PIL
from PIL import Image
import argparse
import os
from tqdm import tqdm

def convert(args):
    name = args.dataset_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    # caption = args.caption
    images = os.listdir(input_dir)
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    for n, image in enumerate(tqdm(images)):

        img = Image.open(os.path.join(input_dir, image)).convert("RGB")
        # basename = os.path.splitext(image)[0]
        caption = input(f"Input caption for {image}:")

        if caption == "":
            continue

        img.save(os.path.join(output_dir, name, f"{name}_{n:04d}.jpg"), quality=98)
        with open(os.path.join(output_dir, name, f"{name}_{n:04d}.txt"), mode='w') as f:
            f.write(caption)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./datasets_with_caption/")
    parser.add_argument("--dataset_name", type=str, required=True)
    # parser.add_argument("--caption", type=str, required=True)

    args = parser.parse_args()
    convert(args)


