from PIL import Image
from resizeimage import resizeimage
import os

for root,dir,files in os.walk('mixed_train/'):
    folder = f'mix_train/{root[12:]}'
    if not os.path.exists(folder):
        os.mkdir(folder)

    for i,file in enumerate(files):
        with open(f'{root}/{file}', 'rb') as f:
            with Image.open(f) as image:
                try:
                    cover = resizeimage.resize_cover(image,[300,300])
                    cover.save(f'{folder}/{file}', image.format)
                except:
                    print(file)
