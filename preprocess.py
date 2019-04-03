from PIL import Image
from resizeimage import resizeimage
import os

reject,take = 0,0
for root,dir,files in os.walk('Images/'):
    print(root)
    train_folder = f'train/{root[10:]}'
    val_folder = f'val/{root[10:]}'
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    split = int(len(files)*0.8)
    for i,file in enumerate(files):
        with open(f'{root}/{file}', 'rb') as f:
            with Image.open(f) as image:
                try:
                    cover = resizeimage.resize_cover(image,[300,300])
                    if i < split:
                        cover.save(f'{train_folder}/{file}', image.format)
                    else:
                        cover.save(f'{val_folder}/{file}', image.format)
                    take+=1
                except:
                    reject+=1
    print (reject,take)
