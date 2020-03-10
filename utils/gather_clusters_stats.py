import numpy as np
import os
from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from keras.datasets import cifar10
from keras.utils import to_categorical
from utils.file import save_to_file
from utils.gmm_utils import find_max, get_cluster_reps, create_cluster_stats


def merge_dicts(*dicts, option=1):
    d = {}
    if option == 1:
        for dict in dicts:
            for key in dict:
                if dict[key] is not None:
                    if key in d:
                        d[key] = np.concatenate((d[key], dict[key]))
                    else:
                        d[key] = dict[key]

    elif option == 2:
        for dict in dicts:
            for key in dict:
                if dict[key] is not None:
                    if key in d:
                        for key2 in dict[key]:
                            d[key][key2] = np.concatenate((d[key][key2], dict[key][key2]))
                    else:
                        d[key] = dict[key]

    return d

SAVING_DIR = pjoin(os.path.abspath(os.getcwd()), 'savings')
WEIGHTS_DIR = pjoin(os.path.abspath(os.getcwd()), 'add_1_weights.09.hdf5')

# --------- GMM parameters
# Choose between 'generative' or 'discriminative' training loss
GMM_training_method = 'discriminative'

# --------- Data parameters
input_shape = (32, 32, 3)

# --------- Model parameters
network_name = 'resnet20'
modeled_layers_name = 'add_1'
n_gaussians = [500]

# --------- Algorithm parameters
# Specify the number of representatives for each cluster to save
n_cluster_reps_to_save = 100
# Equals for number of batches (not batch size!)
num_of_iterations = 20

if not os.path.exists( SAVING_DIR ):
    os.makedirs( SAVING_DIR )

# -----------------------   Prepare cifar 10 dataset    --------------------------
(_, _), (x_val, y_val) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_one_hot = to_categorical(y_val, 10)

# Normalize the data
x_val = x_val.astype('float32')
x_val /= 255

# subtract pixel mean
x_val_mean = np.mean(x_val, axis=0)
x_val -= x_val_mean

n_data = y_one_hot.shape[0]

interval = np.floor_divide(n_data, num_of_iterations)
model = GMM_CNN( n_gaussians=n_gaussians,
                 input_shape=input_shape,
                 n_classes=10,
                 training_method=GMM_training_method,
                 saving_dir=SAVING_DIR,
                 layers_to_model=modeled_layers_name,
                 network_name=network_name,
                 set_classification_layer_as_output=False,
                 set_gmm_activation_layer_as_output=True,
                 set_gmm_layer_as_output=True )

model.build_model()
model.compile_model()

# Loading GMM-CNN weights
model.load_weights_from_file(WEIGHTS_DIR)


clusters_stats_dict = {}
clusters_rep_dict = {}
for gmm_output_layer in model.gmm_dict.values():
    tot_cluster_rep = {}
    tot_clusters_rep_to_save = {}
    for i in range(num_of_iterations):
        print(f'begin iterate {i+1}/{num_of_iterations}')
        if i + 1 == num_of_iterations:
            preds = model.predict(x_val[i * interval:], batch_size=5)
        else:
            preds = model.predict(x_val[i * interval:(i + 1) * interval], batch_size=5)

        gmm_preds = preds['GMM'][gmm_output_layer]
        if model.set_gmm_layer_as_output:
            llr_gmm_preds = preds['GMM']['llr_'+gmm_output_layer]
        else:
            llr_gmm_preds = None
        clusters_rep, clusters_rep_to_save = find_max(gmm_preds, llr_gmm_preds, n_cluster_reps_to_save)
        clusters_rep = clusters_rep[0]
        clusters_rep_to_save = clusters_rep_to_save[0]

        if bool(tot_cluster_rep):
            tot_cluster_rep = merge_dicts(clusters_rep, tot_cluster_rep, option=1)
        else:
            tot_cluster_rep.update(clusters_rep)
        if bool(tot_clusters_rep_to_save):
            tot_clusters_rep_to_save = merge_dicts(clusters_rep_to_save, tot_clusters_rep_to_save, option=2)
        else:
            tot_clusters_rep_to_save.update(clusters_rep_to_save)

    shape = np.shape(gmm_preds)
    n_samples = 0
    if len(shape) == 2:
        n_samples = n_data
        H_dim = None
        W_dim = None
    elif len(shape) == 4:
        B_dim = n_data
        H_dim = shape[1]
        W_dim = shape[2]
        n_samples = B_dim * H_dim * W_dim
    else:
        raise ValueError('the output shape must be either 4 for convolutional layer or'
                         ' 2 for dense layer, but got: {len(shape)}')

    clusters_rep_to_save = get_cluster_reps(tot_clusters_rep_to_save, H_dim, W_dim, n_cluster_reps_to_save)
    clusters_stats = create_cluster_stats(y_one_hot, tot_cluster_rep, n_samples)

    clusters_stats_dict[gmm_output_layer] = clusters_stats
    clusters_rep_dict[gmm_output_layer] = clusters_rep_to_save

    print('finish gathering results for layer ' + gmm_output_layer)

print('saving results')
save_to_file(file_dir=SAVING_DIR,
             objs_name=['clusters_stats', 'clusters_representatives'],
             objs=[clusters_stats_dict, clusters_rep_dict])

