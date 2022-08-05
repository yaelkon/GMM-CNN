import os
from os.path import join as pjoin

import matplotlib.patches as patches
# sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

from gmm_cnn import GMM_CNN
from receptivefield import ReceptiveField
from utils.file import load_from_file
from utils.file import makedir
from utils.vis_utils import crop_image, get_cifar10_labels

EXP_PATH = pjoin(*['C:', os.environ["HOMEPATH"], 'Desktop', 'tmp', 'resnet20_cifar10'])

# Load config
config = load_from_file(EXP_PATH, ['config'])[0]

# Load model
model = GMM_CNN()
model.load_model(config=config)

# Visualization parameters
# How to vis the representatives images: 1. 'rectangle' around the origin image. 2. 'patches' draw only the rf
vis_option = 'rectangle'
label_fontsize = 26
dot_radius = 0.5
n_representatives = 6

# make clusters saving directory
vis_dir = pjoin(EXP_PATH, 'clusters_vis')
makedir(vis_dir)

# get data
(_, _), (X_images, y_val) = cifar10.load_data()
y_one_hot = to_categorical(y_val, 10)
class_labels = get_cifar10_labels()

clusters_rep_name = 'clusters_representatives'
clusters_stats_name = 'clusters_stats'
clusters_rep_path = os.path.join(EXP_PATH, clusters_rep_name+'.json')
clusters_stats_path = os.path.join(EXP_PATH, clusters_stats_name+'.json')

if os.path.isfile(clusters_rep_path) and os.access(clusters_rep_path, os.R_OK):
    # checks if file exists
    print("File exists and is readable")
    print("Load results...")
    clusters_representatives = load_from_file(EXP_PATH, [clusters_rep_name])[0]

else:
    raise ValueError("Either file is missing or is not readable")

if os.path.isfile(clusters_stats_path) and os.access(clusters_stats_path, os.R_OK):
    # checks if file exists
    print("File \'clusters_stats\' exists and is readable")
    print("Load results...")
    clusters_stats = load_from_file(EXP_PATH, [clusters_stats_name])[0]

else:
    raise ValueError("Need to run \'gatherClustersStats\' script first")

rf = ReceptiveField( model.keras_model )

for gmm_name in clusters_representatives:
    fc_flag = False
    gmm_layer = model.keras_model.get_layer(gmm_name)
    layer_name = model.get_correspond_conv_layer(gmm_name)
    if gmm_layer.modeled_layer_type == 'Dense':
        fc_flag = True

    print('cluster vis to layer: ', layer_name)
    layer_dir = os.path.join(vis_dir, layer_name)
    makedir(layer_dir)

    for c in clusters_representatives[gmm_name]:
        # If the cluster is empty -> don't draw it
        if clusters_stats[gmm_name][c] is not None:
            nc_samples = len(clusters_representatives[gmm_name][c]['image'])
            if nc_samples < n_representatives:
                n_rep = nc_samples
            else:
                n_rep = n_representatives

            unique_image_inds, unique_inds = np.unique(clusters_representatives[gmm_name][c]['image'], return_index=True)
            sorted_unique_inds = np.argsort(unique_inds)
            unique_inds = unique_inds[sorted_unique_inds]
            unique_image_inds = unique_image_inds[sorted_unique_inds]

            if len(unique_image_inds) < n_rep:
                unique = False
            else:
                unique = True
            if n_rep < 4:
                n_rows = 1
                n_cols = n_rep
            else:
                n_cols = int(np.ceil(n_rep / 2))
                n_rows = int(np.ceil(n_rep / n_cols))

            fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10,8))

            for j in range(n_rep):
                if unique:
                    index = unique_inds[j]
                else:
                    index = j
                if not fc_flag:
                    target_pos = (clusters_representatives[gmm_name][c]['spatial_location']['row'][index],
                                  clusters_representatives[gmm_name][c]['spatial_location']['col'][index])
                    # add a center dot
                    size, center, upper_left_pos, origin_center = rf.target_neuron_rf( layer_name, target_pos,
                                                                                       rf_layer_name='input_layer',
                                                                                       return_origin_center=True,
                                                                                       return_upper_left_pos=True )
                    # The upper and left rectangle coordinates
                    upper_left_row = upper_left_pos[0]
                    upper_left_col = upper_left_pos[1]

                row = int( np.floor_divide( j, n_cols ) )
                col = int( np.remainder( j, n_cols ) )

                img_ind = clusters_representatives[gmm_name][c]['image'][index]
                img = X_images[img_ind]
                img_label = class_labels[int(np.where(y_one_hot[img_ind] == 1)[0])]

                if fc_flag:
                    if n_cols == 1:
                        ax.imshow(img)
                    elif n_rows == 1:
                        ax[col].imshow(img)
                    else:
                        ax[row, col].imshow(img)

                else:
                    if vis_option == 'rectangle':
                        # Create a center dot patch
                        dot = patches.Circle((origin_center[1], origin_center[0]), radius=dot_radius,
                                             linewidth=1.25, edgecolor='r', facecolor='r')
                        # Create a Rectangle patch
                        rect = patches.Rectangle((upper_left_col, upper_left_row), size[1]-1, size[0]-1,
                                                 linewidth=4, edgecolor='r', facecolor='none')
                        # Add the patch to the Axes
                        if n_cols == 1:
                            ax.imshow(img)
                            ax.add_patch(rect)
                            ax.add_patch(dot)
                        elif n_rows == 1:
                            ax[col].imshow(img)
                            ax[col].add_patch( rect )
                            ax[col].add_patch( dot )
                        else:
                            ax[row, col].imshow(img)
                            ax[row, col].add_patch( rect )
                            ax[row, col].add_patch( dot )

                    elif vis_option == 'patches':
                        new_center_row = origin_center[0] - upper_left_row
                        new_center_col = origin_center[1] - upper_left_col
                        crop_img = crop_image(image=img, row_pixel=upper_left_row, column_pixel=upper_left_col,
                                              height=size[0], width=size[1])
                        # Create a center dot patch
                        dot = patches.Circle((new_center_col, new_center_row), radius=dot_radius,
                                             linewidth=1.25, edgecolor='r', facecolor='r')
                        if n_cols == 1:
                            ax.imshow(crop_img)
                            ax.add_patch(dot)
                        elif n_rows == 1:
                            ax[col].imshow(crop_img)
                            ax[col].add_patch(dot)
                        else:
                            ax[row, col].imshow(crop_img)
                            ax[row, col].add_patch(dot)
                    else:
                        raise ValueError('vision option can be either rectangle or patches')

                if n_cols == 1:
                    ax.set_title(img_label, fontdict={'fontsize': label_fontsize})
                elif n_rows == 1:
                    ax[col].set_title(img_label, fontdict={'fontsize': label_fontsize})
                else:
                    ax[row, col].set_title(img_label, fontdict={'fontsize': label_fontsize})

            if fc_flag:
                fig_dir = os.path.join( layer_dir, 'cluster_' + c + '.png' )
            else:
                fig_dir = os.path.join(layer_dir, 'cluster_' + c + '_' + vis_option + '.png')

            for row in range(n_rows):
                for col in range(n_cols):
                    if n_cols == 1:
                        ax.axis('off')
                    elif n_rows == 1:
                        ax[col].axis('off')
                    else:
                        ax[row, col].axis('off')

            plt.savefig(fig_dir)
            plt.close(fig)