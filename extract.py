import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    
    from osgeo import gdal

    def get_cores(path, results):

        from osgeo import gdal

        slide = gdal.Open(path)

        b1 = slide.GetRasterBand(1)
        b2 = slide.GetRasterBand(2)
        b3 = slide.GetRasterBand(3)
        
        for name, c, r, x, y, w, h, QC_pass in results.itertuples(name=None):
            
            core = np.zeros((h, w, 3), dtype='uint8')
            
            core[..., 0] = b1.ReadAsArray(x, y, w, h)
            core[..., 1] = b2.ReadAsArray(x, y, w, h)
            core[..., 2] = b3.ReadAsArray(x, y, w, h)
        
            yield c, r, Image.fromarray(core)
    
except:
    
    def get_cores(path, results):

        img = Image.open(path).convert('RGB')

        img = np.asarray(img)

        for name, c, r, x, y, w, h, QC_pass in results.itertuples(name=None):

            yield c, r, Image.fromarray(img[y:y+h, x:x+w])



if __name__ == '__main__':
    
    import os
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path',
        type=str,
        default='/path_to_TMA_img/',
        help='TMA image path.'
    )
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=70,
        help='quality of JPEG images to be saved.'
    )

    args, _ = parser.parse_known_args()
    
    results = pd.read_csv(os.path.splitext(args.path)[0] + '.csv', index_col='name')
    
    output_dir = os.path.splitext(args.path)[0]
    os.makedirs(output_dir, exist_ok=True)
    
    for c, r, img in get_cores(args.path, results):
        img.save(os.path.join(output_dir, f'{c}_{r}.jpg'), quality=args.quality)
    