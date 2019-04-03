from PIL import Image
from resizeimage import resizeimage
import os

reject,take = 0,0
for root,dir,files in os.walk('base/'):
    print(root)
    if not os.path.exists('test'):
        os.mkdir('test')

    for i,file in enumerate(files):
        try:
            with open(f'{root}/{file}', 'rb') as f:
                with Image.open(f) as image:
                        cover = resizeimage.resize_cover(image,[300,300])
                        cover.save(f'test/0/{file}', image.format)
        except:
            print('d')
