import os
import glob
import shutil

src_dir = r'D:\PlacSeg\data\To_Be_Annotated_Images\NaNo_training_6_7_2023'
search_root = r'D:\PlacSeg\data\To_Be_Annotated_Images'
search_dirs = glob.glob(os.path.join(search_root, '*'))
search_dirs = [x for x in search_dirs if x != src_dir]

src_imgs = glob.glob(os.path.join(src_dir, '*.png'))
for i, img in enumerate(src_imgs):
    for search_dir in search_dirs:
        search_imgs = glob.glob(os.path.join(search_dir, '*.png'))
        for search_img in search_imgs:
            if os.path.basename(img) == os.path.basename(search_img):
                print(f'Found {img} in {search_dir}')
                shutil.remove(img)
