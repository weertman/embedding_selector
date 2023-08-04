# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:22:26 2023

@author: wlwee
"""

#%%
from embedding_selector import embedding_selector as ES
import os
import glob
import shutil

#%%

path_videos = glob.glob(r'D:\OctoSeg\data\Mitch\*\*\*.mp4')

target_dir = os.path.join(r'D:\OctoSeg\data\Mitch_selected_images')
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)
else:
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    

#%%

imgs = ES.use_embedding_selector_on_videos(target_dir,
                                            path_videos,
                                            n_components_umap=100,
                                            target_number_of_imgs=3000,
                                            nimgs_embedding=500000,
                                            n_clusters=3000,
                                            target_dim=(192,120),
                                            resize=True,
                                            convert_to_gray_for_pca=False,
                                            save_target_dim=(960, 600),
                                            plot=False)









