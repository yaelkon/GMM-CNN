import numpy as np
import os
import glob
from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from keras.datasets import cifar10
from keras.utils import to_categorical
from utils.file import save_to_file, load_from_file
from utils.gmm_utils import find_max, get_cluster_reps, create_cluster_stats
from PIL import Image


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


EXP_PATH = pjoin(*['C:', os.environ["HOMEPATH"], 'Desktop', 'tmp', 'resnet20_cifar10'])
WATERMARK_EXP = False

# --------- Algorithm parameters
# Specify the number of representatives for each cluster to save
n_cluster_reps_to_save = 100
# Equals for number of batches (not batch size!)
num_of_iterations = 5
batch_size = 5
# Get the saved model
# model_dir = os.path.join( EXP_PATH, 'keras_model.hdf5' )

# Load config
config = load_from_file(EXP_PATH, ['config'])[0]
config['set_classification_layer_as_output'] = False
config['set_gmm_activation_layer_as_output'] = True
config['set_gmm_layer_as_output'] = True

list_of_weights = glob.glob( pjoin( EXP_PATH, 'weights.*.hdf5' ) )
weights_dir = list_of_weights[-1]
# Load model
model = GMM_CNN()
model.load_model(weights_dir=weights_dir, config=config)
model.set_gmm_classification_weights()

# -----------------------   Prepare cifar 10 dataset    --------------------------
(_, _), (x_val, y_val) = cifar10.load_data()

# Collect the data for WM experiment
WM_DIR = pjoin(*[EXP_PATH, 'Watermark_Data', 'validation'])
if WATERMARK_EXP and os.path.isdir(WM_DIR):
    print("Loading Watermark data")
    x_val = np.empty_like(x_val)
    list_dir = sorted(os.listdir(WM_DIR))
    for i, l in enumerate(list_dir):
        img_dir = pjoin(WM_DIR, l)
        image = Image.open(img_dir)
        x_val[i] = np.array(image)
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

clusters_stats_dict = {}
clusters_rep_dict = {}

for gmm_layer in model.gmm_dict.values():
    tot_cluster_rep = {}
    tot_clusters_rep_to_save = {}
    for i in range(num_of_iterations):
        print(f'begin iterate {i+1}/{num_of_iterations}')

        if i + 1 == num_of_iterations:
            preds = model.predict(x_val[i * interval:], batch_size=batch_size)
            # res = model.evaluate(preds['classification'], y_one_hot[i * interval:])
            indices = np.arange(i*interval, x_val.shape[0])
        else:
            preds = model.predict(x_val[i * interval:(i + 1) * interval], batch_size=batch_size)
            # res = model.evaluate(preds['classification'], y_one_hot[i * interval:(i + 1) * interval])
            indices = np.arange(i * interval, (i + 1) * interval)

        gmm_preds = preds['GMM'][gmm_layer]
        if model.set_gmm_layer_as_output:
            llr_gmm_preds = preds['GMM']['llr_'+gmm_layer]
        else:
            llr_gmm_preds = None
        clusters_rep, clusters_rep_to_save = find_max(gmm_preds, llr_gmm_preds, n_cluster_reps_to_save,
                                                      indices=indices)
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
        raise ValueError('The output shape must be either 4 for convolutional layer or'
                         ' 2 for dense layer, but got: {len(shape)}')

    clusters_rep_to_save = get_cluster_reps(tot_clusters_rep_to_save, H_dim, W_dim, n_cluster_reps_to_save)
    clusters_stats = create_cluster_stats(y_one_hot, tot_cluster_rep, n_samples)

    clusters_stats_dict[gmm_layer] = clusters_stats
    clusters_rep_dict[gmm_layer] = clusters_rep_to_save

    print('finish gathering results for layer ' + gmm_layer)

print('saving results')
save_to_file(file_dir=EXP_PATH,
             objs_name=['clusters_stats', 'clusters_representatives'],
             objs=[clusters_stats_dict, clusters_rep_dict])
