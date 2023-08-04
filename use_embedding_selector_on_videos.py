# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:36:46 2023

@author: wlwee
"""

from embedding_selector import embedding_selector as ES
import os
import glob
import shutil

path_videos = glob.glob(r'D:\PlacSeg\data\SourceVideos\*\*\*.MOV')
target_dir = os.path.join(r'D:\PlacSeg\data\To_Be_Annotated_Images\NaNo_training_6_7_2023')
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)
else:
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)

imgs = ES.use_embedding_selector_on_videos(target_dir,
                                            path_videos,
                                            n_components_umap=50,
                                            target_number_of_imgs=500,
                                            nimgs_embedding=100000,
                                            n_clusters=500,
                                            target_dim=(192, 108),
                                            resize=True,
                                            convert_to_gray_for_pca=False,
                                            save_target_dim=(960, 540),
                                            plot=False)