import os
import PIL
from PIL import Image
import argparse
import os
from tqdm import tqdm
import random

def convert(args):
    name = args.dataset_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_generate = args.n_generate
    # caption = args.caption
    images = os.listdir(input_dir)
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    width_list = [480, 520, 560, 600]

    for n, image in enumerate(tqdm(images)):

        img = Image.open(os.path.join(input_dir, image)).convert("RGBA")
        crop = img.split()[-1].getbbox()
        img = img.crop(crop)
        w_, h_ = img.size
        caption = input(f"Input caption for {image}:")

        if caption == "":
            continue

        for i in tqdm(range(n_generate)):
            w, h = random.choices(width_list, k=2)
            new_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
            f = random.uniform(0.5, 0.8)
            
            img_edit = img.resize((int(f * w), int(f * w * (h_ / w_))), Image.Resampling.BICUBIC)
            width, height = img_edit.size

            x, y = random.randint(0, int(w - width)), random.randint(0, int(h / 2))
            new_img.paste(img_edit, (x, y))

            datas = new_img.getdata()
            newData = []
            for item in datas:
                if item[3] == 0:
                    newData.append((255, 255, 255, 255))
                else:
                    newData.append(item)
            new_img.putdata(newData)

            new_img.convert("RGB").save(os.path.join(output_dir, name, f"{name}_{n:04d}_{i:04d}.jpg"), quality=98)
            with open(os.path.join(output_dir, name, f"{name}_{n:04d}_{i:04d}.txt"), mode='w') as f:
                f.write(caption)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./datasets_with_caption/")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--n_generate", type=int, default=20)
    # parser.add_argument("--caption", type=str, required=True)

    args = parser.parse_args()
    convert(args)


