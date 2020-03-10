import os
# sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from keras.datasets import cifar10
from keras.utils import to_categorical
from receptivefield import ReceptiveField
from utils.file import load_from_file
from utils.file import makedir
from visualization.utils import crop_image, get_cifar10_labels

ROOT_DIR = os.path.abspath(os.curdir)
SAVING_DIR = pjoin(os.path.abspath(os.getcwd()), 'savings')
WEIGHTS_DIR = pjoin(os.path.abspath(os.getcwd()), 'add_1_weights.09.hdf5')

# Visualization parameters
# How to vis the representatives images: 1. 'rectangle' around the origin image. 2. 'patches' draw only the rf
vis_option = 'patches'
label_fontsize = 26
dot_radius = 1
n_representatives = 6

# make clusters saving directory
vis_dir = pjoin(SAVING_DIR, 'clusters_vis')
makedir(vis_dir)

# --------- GMM parameters
# Choose between 'generative' or 'discriminative' training loss
GMM_training_method = 'discriminative'

# --------- Data parameters
input_shape = (32, 32, 3)

# --------- Model parameters
network_name = 'resnet20'
modeled_layers_name = 'add_1'
n_gaussians = [500]

# get data
(_, _), (X_images, y_val) = cifar10.load_data()
y_one_hot = to_categorical(y_val, 10)
class_labels = get_cifar10_labels()

# Load model
model = GMM_CNN(n_gaussians=n_gaussians,
                input_shape=input_shape,
                n_classes=10,
                training_method=GMM_training_method,
                saving_dir=SAVING_DIR,
                GMM_layers=modeled_layers_name,
                network_name=network_name,
                set_classification_layer_as_output=False,
                set_gmm_activation_layer_as_output=True,
                set_gmm_layer_as_output=True)

model.build_model()
model.compile_model()

# Loading GMM-CNN weights
model.load_weights_from_file(WEIGHTS_DIR)

clusters_rep_name = 'clusters_representatives'
clusters_stats_name = 'clusters_stats'
clusters_rep_path = os.path.join(SAVING_DIR, clusters_rep_name+'.json')
clusters_stats_path = os.path.join(SAVING_DIR, clusters_stats_name+'.json')

if os.path.isfile(clusters_rep_path) and os.access(clusters_rep_path, os.R_OK):
    # checks if file exists
    print("File exists and is readable")
    print("Load results...")
    clusters_representatives = load_from_file(SAVING_DIR, [clusters_rep_name])[0]

else:
    raise ValueError("Either file is missing or is not readable")

if os.path.isfile(clusters_stats_path) and os.access(clusters_stats_path, os.R_OK):
    # checks if file exists
    print("File \'clusters_stats\' exists and is readable")
    print("Load results...")
    clusters_stats = load_from_file(SAVING_DIR, [clusters_stats_name])[0]

else:
    raise ValueError("Need to run \'gatherClustersStats\' script first")

rf = ReceptiveField( model.keras_model )

for gmm_name in clusters_representatives:
    draw_layer = False
    for layer_name, value in model.gmm_dict.items():
        if gmm_name == value:
            if layer_name in model.modeled_layers and 'fc' not in layer_name and 'class' not in layer_name:
                draw_layer = True
                break
    if draw_layer:
        print('cluster vis to layer: ', layer_name)
        layer_dir = os.path.join(vis_dir, layer_name)
        makedir(layer_dir)
        n_clusters = len(clusters_representatives[gmm_name])
        layer_rf_size = rf.size[layer_name]

        for c in clusters_representatives[gmm_name]:
            # If the cluster is empty -> don't draw it
            if clusters_stats[gmm_name][c] is not None:
                    # and clusters_stats[gmm_name][c]['fullness_percent'] > 0:
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
                    # if layer_rf_size[0] >= model.keras_model.input_shape[1]:
                    if unique:
                        index = unique_inds[j]
                    else:
                        index = j
                    target_pos = (clusters_representatives[gmm_name][c]['spatial_location']['row'][index],
                                  clusters_representatives[gmm_name][c]['spatial_location']['col'][index])
                    img_ind = clusters_representatives[gmm_name][c]['image'][index]

                    img = X_images[img_ind]
                    img_label = class_labels[int(np.where(y_one_hot[img_ind] == 1)[0])]

                    # add a center dot
                    size, center, upper_left_pos, origin_center = rf.target_neuron_rf( layer_name, target_pos,
                                                                                       rf_layer_name='input_layer',
                                                                                       return_origin_center=True,
                                                                                       return_upper_left_pos=True )
                    dot = patches.Circle((origin_center[1], origin_center[0]), radius=dot_radius,
                                         linewidth=1.25, edgecolor='r', facecolor='r')

                    # The upper and left rectangle coordinates
                    upper_left_row = upper_left_pos[0]
                    upper_left_col = upper_left_pos[1]
                    row = int(np.floor_divide(j, n_cols))
                    col = int(np.remainder(j, n_cols))

                    if vis_option == 'rectangle':
                        if n_cols == 1:
                            ax.imshow(img)
                        elif n_rows == 1:
                            ax[col].imshow(img)
                        else:
                            ax[row, col].imshow(img)
                        # Create a Rectangle patch
                        rect = patches.Rectangle((upper_left_col, upper_left_row), size[1]-1, size[0]-1,
                                                 linewidth=4, edgecolor='r', facecolor='none')

                        # Add the patch to the Axes
                        if n_cols == 1:
                            ax.add_patch(rect)
                            ax.add_patch(dot)
                        elif n_rows == 1:
                            ax[col].add_patch(rect)
                            ax[col].add_patch(dot)
                        else:
                            ax[row, col].add_patch(rect)
                            ax[row, col].add_patch(dot)

                    elif vis_option == 'patches':
                        crop_img = crop_image(image=img, row_pixel=upper_left_row, column_pixel=upper_left_col,
                                              height=size[0], width=size[1])
                        if n_cols == 1:
                            ax.imshow(crop_img)
                        elif n_rows == 1:
                            ax[col].imshow(crop_img)
                        else:
                            ax[row, col].imshow(crop_img)

                    if n_cols == 1:
                        ax.set_title(img_label, fontdict={'fontsize': label_fontsize})
                    elif n_rows == 1:
                        ax[col].set_title(img_label, fontdict={'fontsize': label_fontsize})
                    else:
                        ax[row, col].set_title(img_label, fontdict={'fontsize': label_fontsize})

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