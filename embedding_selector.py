
import umap
import glob
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
import sklearn.cluster as cluster

class embedding_selector ():
    
    def use_embedding_selector_on_images (target_dir, path_imgs, target_number_of_images, 
                                          n_clusters, n_components_pca=50, n_components_umap=2,
                                          n_neighbors=15, min_dist=0.1, metric='euclidean',
                                          convert_to_gray_for_pca = False,
                                          save_imgs = True, resize=True, target_dim=(50,50), 
                                          resize_save_imgs = True, save_target_dim=(640,640), 
                                          plot=True, random_state=42):
        
        def given_path_imgs_return_imgs (path_imgs, convert_to_gray_for_pca, target_dim):
                imgs = []
                pbar = tqdm(total = len(path_imgs), position=0, leave=True)
                for path_img in path_imgs:
                    img = cv2.imread(path_img)
                    img = cv2.resize(img, target_dim)
                    if convert_to_gray_for_pca == True:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgs.append(img)
                    pbar.update(n=1)
                pbar.close()
                imgs = np.array(imgs)
                return imgs

        def given_imgs_return_umap_embedding_of_pca (imgs, n_components_pca, n_components_umap,
                                                    n_neighbors, min_dist, metric, convert_to_gray_for_pca):
            print('Creating PCA embedding...')
            def given_imgs_return_princple_components(imgs, n_components, convert_to_gray_for_pca):
                imgs = imgs.copy()
                if convert_to_gray_for_pca == True:
                    imgs = np.array([s.flatten() for s in imgs])
                else:
                    imgs_tmp = imgs.copy()
                    imgs = []
                    for i, im in enumerate(imgs_tmp):
                        r,g,b = cv2.split(im)
                        r,g,b = r.flatten(), g.flatten(), b.flatten()
                        im = np.hstack([r,g,b])
                        imgs.append(im)
                    imgs = np.array(imgs)
                pca = PCA(n_components=n_components)
                comps = pca.fit(imgs.T)
                pca_img_comps = np.array(comps.components_).T
                return pca_img_comps
            pca_img_comps = given_imgs_return_princple_components(imgs, n_components_pca,  convert_to_gray_for_pca)
            print('PCA components shape =', pca_img_comps.shape)

            if pca_img_comps.shape[1] > imgs.shape[0]:
                print(f'Warning: n_components_pca ({n_components_pca}) is greater than the number of images ({imgs.shape[0]}).')
                print(f'         This will result in a failure of embedding.')

            print('Creating UMAP embedding...')
            reducer = umap.UMAP(
                n_neighbors = n_neighbors,
                n_components = n_components_umap,
                min_dist=min_dist,
                metric=metric,
                verbose=True
            )
            embedding = reducer.fit_transform(pca_img_comps)
            print('UMAP embedding created.')
            print('Embedding shape =', embedding.shape)

            return embedding
        
        def given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_imgs, 
                                                                n_clusters, random_state):
            print('Clustering embedding...')

            nimgs = len(embedding)
            imgs_per_cluster = int(nimgs/n_clusters)

            print('nimgs =', nimgs)
            print('target_number_of_imgs =', target_number_of_imgs)
            print('imgs in a cluster =', imgs_per_cluster)

            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(embedding)

            print('Clustering complete.')
            clusters = sorted([s for s in set(kmeans.labels_.tolist())])
            print('Number of clusters =', len(clusters))
            print('Clusters =', clusters)

            return kmeans.labels_

        def create_pair_plot_kmeans (embedding, kmeans_labels, target_dir):
            print('Creating pair plot...')
            df = pd.DataFrame(embedding)
            df['labels'] = kmeans_labels
            sns.pairplot(df, hue="labels", markers='.', corner=True, palette = 'cubehelix')
            plt.savefig(os.path.join(target_dir, 'pair_plot.png'))
        
        def given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, 
                                                        target_number_of_images, target_dim, plot):
            print('Clustering and grabbing keep indexes...')
            
            keep_idxs = []
            clusters = sorted([s for s in set(kmeans_labels.tolist())])
            imgs_per_cluster = int(target_number_of_images/len(clusters))

            print(f'Grabbing {imgs_per_cluster} from {len(clusters)}')
            print(f'Grabbing from {imgs.shape}')

            if plot == True:
                fscalar = 1
                fig, axs = plt.subplots(imgs_per_cluster, len(clusters),
                                        figsize = (int(fscalar*len(clusters)),
                                                int(fscalar*imgs_per_cluster)))
            pbar = tqdm(total = int(len(clusters) * imgs_per_cluster),
                        position=0, leave=True)
            for i, clst in enumerate(clusters):
                idxs = np.where(kmeans_labels == clst)[0].tolist()
                ridxs = random.sample(idxs, imgs_per_cluster)
                
                for j, idx in enumerate(ridxs):
                    if plot == True:
                        im = imgs[idx]
                        if imgs_per_cluster == 1:
                            ax = axs[i]
                        else:
                            ax = axs[j,i]
                        ax.imshow(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                        ax.axis('off')
                    keep_idxs.append(idx)
                    
                    if j == 0:
                        if plot == True:
                            ax.set_title(clst)
                    
                    pbar.update(n=1)
            pbar.close()

            if plot == True:
                fig.tight_layout()
                if imgs_per_cluster > 3:
                    fig_name = 'UMAP_Selection_facetGrid.png'
                    fig_path = os.path.join(target_dir, fig_name)
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                else:
                    print(f'Not saving figure because we are grabbing imgs_per_cluster = {imgs_per_cluster} which is < 4 and it will look bad/explode')
                plt.show(fig)
                plt.close(fig)

            print('Keep indexes grabbed.')
            print('Number of keep indexes =', len(keep_idxs))
            print('Keep indexes =', keep_idxs)

            return keep_idxs

        def given_keep_idxs_return_imgs(path_imgs, keep_idxs, target_dir, save_imgs, save_target_dim, resize_save_imgs):
            print('Grabbing keep images...')
            keep_imgs = []
            pbar = tqdm(total = len(keep_idxs), position=0, leave=True)
            for idx in keep_idxs:
                im_path = path_imgs[idx]
                im_name = os.path.basename(im_path)
                im = cv2.imread(im_path)
                if save_imgs == True:
                    if resize_save_imgs == True:
                        im = cv2.resize(im, save_target_dim)
                    new_im_path = os.path.join(target_dir, im_name)
                    cv2.imwrite(new_im_path, im)
                keep_imgs.append(im)
                pbar.update(n=1)
            pbar.close()
            keep_imgs = np.array(keep_imgs)
            print('Keep images grabbed.')
            print('Keep images =', keep_imgs.shape)

            return keep_imgs
        
        imgs = given_path_imgs_return_imgs(path_imgs, convert_to_gray_for_pca, target_dim)

        embedding = given_imgs_return_umap_embedding_of_pca(imgs, n_components_pca, n_components_umap, n_neighbors, 
                                                            min_dist, metric, convert_to_gray_for_pca)
        kmeans_labels = given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_images, 
                                                                            n_clusters, random_state)
        if plot == True:
            create_pair_plot_kmeans(embedding, kmeans_labels, target_dir)
        keep_idxs = given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, target_number_of_images, 
                                                            target_dim, plot)
        keep_imgs = given_keep_idxs_return_imgs(path_imgs, keep_idxs, target_dir, save_imgs, save_target_dim, resize_save_imgs)

        return keep_imgs

    def use_embedding_selector_on_videos (target_dir, videos, target_number_of_imgs, 
                                        nimgs_embedding, n_clusters,
                                        n_components_pca=50, n_components_umap=2, 
                                        n_neighbors=15, min_dist=0.1, metric='euclidean',
                                        convert_to_gray_for_pca = False,
                                        save_imgs = True, resize=True, target_dim=(50,50), 
                                        save_target_dim=(640,640), plot=True,
                                        random_state=42):

        def given_video_list_upper_limit_return_subset_by(videos, upper_limit):

            def given_video_list_count_imgs(videos):
                total_imgs = 0
                pbar = tqdm(total = len(videos), position=0, leave=True)
                for video in videos:
                    cap = cv2.VideoCapture(video)
                    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_imgs += l
                    cap.release()
                    pbar.update(1)
                pbar.close()
                return total_imgs

            def given_upper_limit_calculate_subset_by(total_imgs, upper_limit):
                return int(total_imgs/upper_limit)

            total_imgs = given_video_list_count_imgs(videos)
            return given_upper_limit_calculate_subset_by(total_imgs, upper_limit)

        def given_video_list_grab_images(videos, subset_by, resize, convert_to_gray_for_pca, target_dim):

            def resize_imgs(imgs, target_dim, convert_to_gray_for_pca):
                resized_imgs = []
                for img in imgs:
                    if convert_to_gray_for_pca == True:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized_imgs.append(cv2.resize(img, target_dim))
                return resized_imgs

            imgs = []
            video_key = {}
            look_up_list = []
            pbar = tqdm(total = len(videos), position=0, leave=True)
            for n, video in enumerate(videos):
                print(video)
                video_key[n] = video
                imgs_tmp = []
                cap = cv2.VideoCapture(video)
                l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(0, l, subset_by):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret == True:
                        imgs_tmp.append(frame)
                        look_up_list.append([n, i])
                    else:
                        print(f'Error reading frame {i} from video {video}')

                cap.release()
                if resize == True:
                    imgs_tmp = resize_imgs(imgs_tmp, target_dim, convert_to_gray_for_pca)
                imgs = imgs + imgs_tmp

                pbar.update(1)
            pbar.close()
            imgs = np.array(imgs)

            return imgs, video_key, look_up_list

        def given_imgs_return_umap_embedding_of_pca (imgs, n_components_pca, n_components_umap,
                                                    n_neighbors, min_dist, metric, convert_to_gray_for_pca):

            def given_imgs_return_princple_components(imgs, n_components, convert_to_gray_for_pca):
                imgs = imgs.copy()
                if convert_to_gray_for_pca == True:
                    imgs = np.array([s.flatten() for s in imgs])
                else:
                    imgs_tmp = imgs.copy()
                    imgs = []
                    for i, im in enumerate(imgs_tmp):
                        r,g,b = cv2.split(im)
                        r,g,b = r.flatten(), g.flatten(), b.flatten()
                        im = np.hstack([r,g,b])
                        imgs.append(im)
                    imgs = np.array(imgs)
                pca = PCA(n_components=n_components)
                comps = pca.fit(imgs.T)
                pca_img_comps = np.array(comps.components_).T
                return pca_img_comps

            pca_img_comps = given_imgs_return_princple_components(imgs, n_components_pca,  convert_to_gray_for_pca)

            print('PCA components shape =', pca_img_comps.shape)

            if pca_img_comps.shape[1] > imgs.shape[0]:
                print(f'Warning: n_components_pca ({n_components_pca}) is greater than the number of images ({imgs.shape[0]}).')
                print(f'         This will result in a failure of embedding.')

            print('Creating UMAP embedding...')
            reducer = umap.UMAP(
                n_neighbors = n_neighbors,
                n_components = n_components_umap,
                min_dist=min_dist,
                metric=metric,
                verbose=True
            )
            embedding = reducer.fit_transform(pca_img_comps)
            print('UMAP embedding created.')
            print('Embedding shape =', embedding.shape)

            return embedding

        def given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_imgs, 
                                                                n_clusters, random_state):
            print('Clustering embedding...')

            nimgs = len(embedding)
            imgs_per_cluster = int(nimgs/n_clusters)

            print('nimgs =', nimgs)
            print('target_number_of_imgs =', target_number_of_imgs)
            print('imgs_per_cluster =', imgs_per_cluster)

            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(embedding)

            print('Clustering complete.')
            clusters = sorted([s for s in set(kmeans.labels_.tolist())])
            print('Number of clusters =', len(clusters))
            print('Clusters =', clusters)

            return kmeans.labels_

        def create_pair_plot_kmeans (embedding, kmeans_labels, target_dir):
            print('Creating pair plot...')
            df = pd.DataFrame(embedding)
            df['labels'] = kmeans_labels
            sns.pairplot(df, hue="labels", markers='.', corner=True, palette = 'cubehelix')
            plt.savefig(os.path.join(target_dir, 'pair_plot.png'))

        def given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, 
                                                        target_number_of_images, target_dim, plot, resize):
            print('Clustering and grabbing keep indexes...')
            
            keep_idxs = []
            clusters = sorted([s for s in set(kmeans_labels.tolist())])
            imgs_per_cluster = int(target_number_of_images/len(clusters))

            print(f'Grabbing {imgs_per_cluster} from {len(clusters)}')
            print(f'Grabbing from {imgs.shape}')

            if plot == True:
                fscalar = 1
                fig, axs = plt.subplots(imgs_per_cluster, len(clusters),
                                        figsize = (int(fscalar*len(clusters)),
                                                int(fscalar*imgs_per_cluster)))
            pbar = tqdm(total = int(len(clusters) * imgs_per_cluster),
                        position=0, leave=True)
            for i, clst in enumerate(clusters):
                idxs = np.where(kmeans_labels == clst)[0].tolist()
                ridxs = random.sample(idxs, imgs_per_cluster)
                
                for j, idx in enumerate(ridxs):
                    if plot == True:
                        im = imgs[idx]
                        if resize == True:
                            im = cv2.resize(im, target_dim)
                        if imgs_per_cluster == 1:
                            ax = axs[i]
                        else:
                            ax = axs[j,i]
                        ax.imshow(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                        ax.axis('off')
                    keep_idxs.append(idx)
                    
                    if j == 0:
                        if plot == True:
                            ax.set_title(clst)
                    
                    pbar.update(n=1)
            pbar.close()

            if plot == True:
                fig.tight_layout()
                if imgs_per_cluster > 3:
                    fig_name = 'UMAP_Selection_facetGrid.png'
                    fig_path = os.path.join(target_dir, fig_name)
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                else:
                    print(f'Not saving figure because imgs_per_cluster = {imgs_per_cluster} < 4. It will look bad')
                plt.show(fig)
                plt.close(fig)

            print('Keep indexes grabbed.')
            print('Number of keep indexes =', len(keep_idxs))
            print('Keep indexes =', keep_idxs)

            return keep_idxs

        def given_keep_idxs_look_up_list_and_video_key_return_imgs (target_dir, keep_idxs, look_up_list, video_key, save_imgs, 
                                                                    resize, save_target_dim):
            print('Grabbing images...')
            
            imgs = []
            pbar = tqdm(total = len(keep_idxs), position=0, leave=True)
            for idx in keep_idxs:
                vid_idx, frame_idx = look_up_list[idx]
                video = video_key[vid_idx]
                cap = cv2.VideoCapture(video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret == True:
                    if resize == True:
                        frame = cv2.resize(frame, save_target_dim)
                    imgs.append(frame)
                    if save_imgs == True:
                        img_name = os.path.basename(video_key[vid_idx]).split('.')[0]
                        img_name = f'{img_name}_img_{idx}.png'
                        img_path = os.path.join(target_dir, img_name)
                        cv2.imwrite(img_path, frame)
                else:
                    print(f'Error reading frame {frame_idx} from video {video}')
                cap.release()
                pbar.update(n=1)
            pbar.close()

            imgs = np.array(imgs)
            print('Images grabbed.')
            print('Images shape =', imgs.shape)

            return imgs

        ## Main
        subset_by = given_video_list_upper_limit_return_subset_by(videos, nimgs_embedding)
        print(f'Using an image every {subset_by} frame to subset the videos to create embedding')
        imgs, video_key, look_up_list = given_video_list_grab_images(videos, subset_by, resize, convert_to_gray_for_pca, target_dim)
        
        print(f'Using {imgs.shape} images to create embedding')
        print(f'Using {len(video_key)} videos to create embedding')

        embedding = given_imgs_return_umap_embedding_of_pca(imgs, n_components_pca, n_components_umap, n_neighbors, 
                                                            min_dist, metric, convert_to_gray_for_pca)
        kmeans_labels = given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_imgs, 
                                                                            n_clusters, random_state)
        if plot == True:
            create_pair_plot_kmeans(embedding, kmeans_labels, target_dir)
        keep_idxs = given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, target_number_of_imgs, 
                                                            target_dim, plot, resize)
        imgs = given_keep_idxs_look_up_list_and_video_key_return_imgs(target_dir, keep_idxs, look_up_list, 
                                                                        video_key, save_imgs, resize, save_target_dim)
        return imgs
    
    def use_embedding_selector_on_videos_returnclips (target_dir, videos, target_number_of_imgs, 
                                                        nimgs_embedding, n_clusters, clip_length = 3,
                                                        n_components_pca=50, n_components_umap=2, 
                                                        n_neighbors=15, min_dist=0.1, metric='euclidean',
                                                        convert_to_gray_for_pca = False,
                                                        save_imgs = True, resize=True, target_dim=(50,50), 
                                                        save_target_dim=(640,640), plot=True,
                                                        random_state=42):

        def given_video_list_upper_limit_return_subset_by(videos, upper_limit):

            def given_video_list_count_imgs(videos):
                total_imgs = 0
                pbar = tqdm(total = len(videos), position=0, leave=True)
                for video in videos:
                    cap = cv2.VideoCapture(video)
                    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_imgs += l
                    cap.release()
                    pbar.update(1)
                pbar.close()
                return total_imgs

            def given_upper_limit_calculate_subset_by(total_imgs, upper_limit):
                return int(total_imgs/upper_limit)

            total_imgs = given_video_list_count_imgs(videos)
            return given_upper_limit_calculate_subset_by(total_imgs, upper_limit)

        def given_video_list_grab_images(videos, subset_by, resize, convert_to_gray_for_pca, target_dim, clip_length):

            def resize_imgs(imgs, target_dim, convert_to_gray_for_pca):
                resized_imgs = []
                for img in imgs:
                    if convert_to_gray_for_pca == True:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized_imgs.append(cv2.resize(img, target_dim))
                return resized_imgs

            imgs = []
            video_key = {}
            look_up_list = []
            pbar = tqdm(total = len(videos), position=0, leave=True)
            for n, video in enumerate(videos):
                print(video)
                video_key[n] = video
                imgs_tmp = []
                cap = cv2.VideoCapture(video)
                l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(clip_length, l-clip_length, subset_by):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret == True:
                        imgs_tmp.append(frame)
                        look_up_list.append([n, i])
                    else:
                        print(f'Error reading frame {i} from video {video}')

                cap.release()
                if resize == True:
                    imgs_tmp = resize_imgs(imgs_tmp, target_dim, convert_to_gray_for_pca)
                imgs = imgs + imgs_tmp

                pbar.update(1)
            pbar.close()
            imgs = np.array(imgs)

            return imgs, video_key, look_up_list

        def given_imgs_return_umap_embedding_of_pca (imgs, n_components_pca, n_components_umap,
                                                    n_neighbors, min_dist, metric, convert_to_gray_for_pca):

            def given_imgs_return_princple_components(imgs, n_components, convert_to_gray_for_pca):
                imgs = imgs.copy()
                if convert_to_gray_for_pca == True:
                    imgs = np.array([s.flatten() for s in imgs])
                else:
                    imgs_tmp = imgs.copy()
                    imgs = []
                    for i, im in enumerate(imgs_tmp):
                        r,g,b = cv2.split(im)
                        r,g,b = r.flatten(), g.flatten(), b.flatten()
                        im = np.hstack([r,g,b])
                        imgs.append(im)
                    imgs = np.array(imgs)
                pca = PCA(n_components=n_components)
                comps = pca.fit(imgs.T)
                pca_img_comps = np.array(comps.components_).T
                return pca_img_comps

            pca_img_comps = given_imgs_return_princple_components(imgs, n_components_pca,  convert_to_gray_for_pca)

            print('PCA components shape =', pca_img_comps.shape)

            if pca_img_comps.shape[1] > imgs.shape[0]:
                print(f'Warning: n_components_pca ({n_components_pca}) is greater than the number of images ({imgs.shape[0]}).')
                print('         This will result in a failure of embedding.')

            print('Creating UMAP embedding...')
            reducer = umap.UMAP(
                n_neighbors = n_neighbors,
                n_components = n_components_umap,
                min_dist=min_dist,
                metric=metric,
                verbose=True
            )
            embedding = reducer.fit_transform(pca_img_comps)
            print('UMAP embedding created.')
            print('Embedding shape =', embedding.shape)

            return embedding

        def given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_imgs, 
                                                                n_clusters, random_state):
            print('Clustering embedding...')

            nimgs = len(embedding)
            imgs_per_cluster = int(nimgs/n_clusters)

            print('nimgs =', nimgs)
            print('target_number_of_imgs =', target_number_of_imgs)
            print('imgs_per_cluster =', imgs_per_cluster)

            kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(embedding)

            print('Clustering complete.')
            clusters = sorted([s for s in set(kmeans.labels_.tolist())])
            print('Number of clusters =', len(clusters))
            print('Clusters =', clusters)

            return kmeans.labels_

        def create_pair_plot_kmeans (embedding, kmeans_labels, target_dir):
            print('Creating pair plot...')
            df = pd.DataFrame(embedding)
            df['labels'] = kmeans_labels
            sns.pairplot(df, hue="labels", markers='.', corner=True, palette = 'cubehelix')
            plt.savefig(os.path.join(target_dir, 'pair_plot.png'))

        def given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, 
                                                     target_number_of_images, target_dim, plot, resize, clip_length):
            print('Clustering and grabbing keep indexes...')
            
            keep_idxs = []
            clusters = sorted([s for s in set(kmeans_labels.tolist())])
            imgs_per_cluster = int(target_number_of_images/len(clusters))

            print(f'Grabbing {imgs_per_cluster} from {len(clusters)}')
            print(f'Grabbing from {imgs.shape}')

            if plot == True:
                fscalar = 1
                fig, axs = plt.subplots(imgs_per_cluster, len(clusters),
                                        figsize = (int(fscalar*len(clusters)),
                                                int(fscalar*imgs_per_cluster)))
            pbar = tqdm(total = int(len(clusters) * imgs_per_cluster),
                        position=0, leave=True)
            for i, clst in enumerate(clusters):
                idxs = np.where(kmeans_labels == clst)[0].tolist()
                ridxs = random.sample(idxs, imgs_per_cluster)
                
                for j, idx in enumerate(ridxs):
                    idx += clip_length
                    if plot == True:
                        im = imgs[idx]
                        if resize == True:
                            im = cv2.resize(im, target_dim)
                        if imgs_per_cluster == 1:
                            ax = axs[i]
                        else:
                            ax = axs[j,i]
                        ax.imshow(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                        ax.axis('off')
                    
                    for c in range(idx-int(clip_length/2), idx+int(clip_length/2)+1):
                        keep_idxs.append(c)
                    
                    if j == 0:
                        if plot == True:
                            ax.set_title(clst)
                    
                    pbar.update(n=1)
            pbar.close()

            if plot == True:
                fig.tight_layout()
                if imgs_per_cluster > 3:
                    fig_name = 'UMAP_Selection_facetGrid.png'
                    fig_path = os.path.join(target_dir, fig_name)
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                else:
                    print(f'Not saving figure because imgs_per_cluster = {imgs_per_cluster} < 4. It will look bad')
                plt.show(fig)
                plt.close(fig)

            print('Keep indexes grabbed.')
            print('Number of keep indexes =', len(keep_idxs))
            print('Keep indexes =', keep_idxs)
            
            keep_idxs = sorted([s for s in set(keep_idxs)])

            return keep_idxs

        def given_keep_idxs_look_up_list_and_video_key_return_imgs (target_dir, keep_idxs, look_up_list, video_key, save_imgs, 
                                                                    resize, save_target_dim):
            print('Grabbing images...')
            
            imgs = []
            pbar = tqdm(total = len(keep_idxs), position=0, leave=True)
            n = 0
            for idx in keep_idxs:
                vid_idx, frame_idx = look_up_list[idx]
                video = video_key[vid_idx]
                cap = cv2.VideoCapture(video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret == True:
                    if resize == True:
                        frame = cv2.resize(frame, save_target_dim)
                    imgs.append(frame)
                    if save_imgs == True:
                        img_name = os.path.basename(video_key[vid_idx]).split('.')[0]
                        img_name = f'{idx}___{img_name}_img.png'
                        img_path = os.path.join(target_dir, img_name)
                        cv2.imwrite(img_path, frame)
                        n += 1
                else:
                    print(f'Error reading frame {frame_idx} from video {video}')
                cap.release()
                pbar.update(n=1)
            pbar.close()

            imgs = np.array(imgs)
            print('Images grabbed.')
            print('Images shape =', imgs.shape)

            return imgs

        ## Main
        subset_by = given_video_list_upper_limit_return_subset_by(videos, nimgs_embedding)
        print(f'Using an image every {subset_by} frame to subset the videos to create embedding')
        imgs, video_key, look_up_list = given_video_list_grab_images(videos, subset_by, resize, convert_to_gray_for_pca, target_dim, clip_length)
        
        print(f'Using {imgs.shape} images to create embedding')
        print(f'Using {len(video_key)} videos to create embedding')

        embedding = given_imgs_return_umap_embedding_of_pca(imgs, n_components_pca, n_components_umap, n_neighbors, 
                                                            min_dist, metric, convert_to_gray_for_pca)
        kmeans_labels = given_embedding_return_kmeans_clustered_embedding(embedding, target_number_of_imgs, 
                                                                            n_clusters, random_state)
        if plot == True:
            create_pair_plot_kmeans(embedding, kmeans_labels, target_dir)
        keep_idxs = given_kmeans_labels_and_return_keep_idxs(target_dir, imgs, kmeans_labels, target_number_of_imgs, 
                                                            target_dim, plot, resize, clip_length)
        imgs = given_keep_idxs_look_up_list_and_video_key_return_imgs(target_dir, keep_idxs, look_up_list, 
                                                                        video_key, save_imgs, resize, save_target_dim)
        return imgs





