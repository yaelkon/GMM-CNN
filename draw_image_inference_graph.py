import glob
import scipy.io as sio
import os
import numpy as np

from keras.datasets import cifar10
from os.path import join as pjoin
from utils.file import load_from_file, makedir
from utils.IG_utils import gather_coOccurrance_mat, get_IG_connections_dict_LR, \
    create_clusters_for_imageGraph, create_connections_heat_maps
from receptivefield import ReceptiveField
from utils.vis_utils import get_cifar10_labels
from layers import get_layer_output
from inference_graph import Inference_Graph
from gmm_cnn import GMM_CNN

# Parameters
EXP_PATH = pjoin(*['C:', os.environ["HOMEPATH"], 'Desktop', 'tmp', 'resnet20_cifar10'])
good_or_bad_image = 'good'
n_image_graphs = 1

# Specify the layers to visualize and the visualization technique for each layer ('rectangle'/'patches')
vis_option = {'add_2': 'rectangle',
              'add_4': 'rectangle',
              'add_6': 'rectangle',
              'add_8': 'rectangle'
              }

n_nodes_in_graph = 3
edge_label_font_size = '22'
header_font_size = '40'
add_heat_map_on_connections = True
calc_heat_map_on_connections = False

# Get the saved weights
list_of_weights = glob.glob( pjoin( EXP_PATH, 'weights.*.hdf5' ) )
weights_dir = list_of_weights[-1]
# Load config
config = load_from_file(EXP_PATH, ['config'])[0]
# Update model config
config['set_gmm_activation_layer_as_output'] = True
# Load model
model = GMM_CNN()
model.load_model(weights_dir=weights_dir, config=config)
model.set_gmm_classification_weights()

layers_to_inference = []
for key in vis_option:
    layers_to_inference.append(key)
layers_to_inference.append(model.keras_model.output_names[0])
# make clusters saving directory
vis_dir = pjoin(EXP_PATH, 'clusters_vis')

# get data
(_, _), (x_val, y_val) = cifar10.load_data()
vis_data = x_val.copy()

# Normalize the data
x_val = x_val.astype('float32')
x_val /= 255

# subtract pixel mean
x_val_mean = np.mean(x_val, axis=0)
x_val -= x_val_mean

classes_labels = get_cifar10_labels()

results_dir = pjoin(EXP_PATH, 'inferences_graphs')
clusters_dir = pjoin(EXP_PATH, 'clusters_vis')
heat_map_dir = pjoin(results_dir, 'heat_maps')
# Create file for results
saving_dir = pjoin(results_dir, 'image_tree')
images_clusters_dir = pjoin(results_dir, 'images_clusters')
makedir(saving_dir)

RF = ReceptiveField(model.keras_model)

layers_clusters_dict = {}
for layer_name in layers_to_inference:
    gmm_name = model.get_correspond_gmm_layer( layer_name )
    n_clusters = model.keras_model.get_layer( gmm_name ).n_clusters
    layers_clusters_dict[layer_name] = n_clusters

test_pred = get_layer_output(keras_model=model.keras_model,
                             layer_name=layers_to_inference[-1],
                             input_data=x_val)
preds_ind = np.argmax(test_pred, axis=1)
preds_confident = np.max(test_pred, axis=1)
y_inds = np.argmax(y_val, axis=1)

if good_or_bad_image == 'good':
    selected_inds = np.where(preds_ind == y_inds)[0]
elif good_or_bad_image == 'bad':
    selected_inds = np.where(preds_ind != y_inds)[0]
else:
    ValueError(f'{good_or_bad_image} should be either \'good\' or \'bad\'')

selected_conf = preds_confident[selected_inds]
sort_selected_args = np.argsort(-selected_conf)

sorted_selected_inds = selected_inds[sort_selected_args]
sorted_selected_conf = selected_conf[sort_selected_args]

selected_false_y = preds_ind[sorted_selected_inds]
selected_true_y = y_inds[sorted_selected_inds]

# Load the global appearance matrices
global_appearance_hist_dict = {}
explained_layer = layers_to_inference[-1]
calc_global_flag = False

for explaining_layer in reversed(layers_to_inference[:-1]):
    key_name = explained_layer + '-' + explaining_layer
    mat_name = pjoin(*[EXP_PATH, 'results', key_name + '.mat'])

    if os.path.isfile(mat_name) and os.access(mat_name, os.R_OK):
        global_appearance_hist_dict[key_name] = sio.loadmat(mat_name)[key_name]
    else:
        calc_global_flag = True
        break
    explained_layer = explaining_layer
# Calc and save global appearance matrices if not exist
if calc_global_flag:
    global_appearance_hist_dict = gather_coOccurrance_mat(gmm_model=model,
                                                          exp_path=EXP_PATH,
                                                          RF=RF,
                                                          layers_to_inference=layers_to_inference,
                                                          data=x_val,
                                                          num_of_iterations=1,
                                                          saving_flag=True,
                                                          return_flag=True)

for img_ind in sorted_selected_inds[:n_image_graphs]:
    img_loc = np.where(sorted_selected_inds == img_ind)[0][0]
    img_pred_conf = sorted_selected_conf[img_loc]
    img_clusters_dir = pjoin(images_clusters_dir, str(img_ind) + '_' + str(round(img_pred_conf, 3)))

    pred_class_index = selected_false_y[img_loc]
    true_class_index = selected_true_y[img_loc]

    expand_img = np.expand_dims(x_val[img_ind], axis=0)
    vis_img = vis_data[img_ind]
    head_class = classes_labels[pred_class_index]
    true_class = classes_labels[true_class_index]

    IT_name = 'inference_image_' + str(img_ind) + '_graph'

    # Calc local appearance matrices
    local_appearance_hist_dict = gather_coOccurrance_mat(gmm_model=model,
                                                         exp_path=EXP_PATH,
                                                         RF=RF,
                                                         layers_to_inference=layers_to_inference,
                                                         data=expand_img,
                                                         num_of_iterations=1,
                                                         saving_flag=False,
                                                         return_flag=True)

    print(f'collecting connections for class {head_class} with image {img_ind}')
    connections_dict = get_IG_connections_dict_LR(class_index=pred_class_index,
                                                  global_appearance_hist_dict=global_appearance_hist_dict,
                                                  local_appearance_hist_dict=local_appearance_hist_dict,
                                                  layers_to_inference=layers_to_inference,
                                                  n_nodes=n_nodes_in_graph)

    if os.path.exists(img_clusters_dir):
        # checks if file exists
        print(f"Clusters for image {img_ind} exists")

    else:
        create_clusters_for_imageGraph(vis_image=vis_img,
                                       image_array=expand_img,
                                       image_false_label=pred_class_index,
                                       true_label=true_class,
                                       gmm_model=model,
                                       rf=RF,
                                       cluster_type=vis_option,
                                       connections_dict=connections_dict,
                                       clusters_dir=clusters_dir,
                                       saving_dir=img_clusters_dir)
    if calc_heat_map_on_connections:
        create_connections_heat_maps(gmm_model=model,
                                     exp_path=EXP_PATH,
                                     RF=RF,
                                     data=x_val,
                                     connections_dict=connections_dict,
                                     saving_dir=heat_map_dir)

    IT = Inference_Graph(name=IT_name,
                         head_class_name=head_class,
                         head_class_index=pred_class_index,
                         connections_dict=connections_dict,
                         node_type=vis_option,
                         n_nodes=n_nodes_in_graph,
                         edge_label_font_size=edge_label_font_size,
                         header_font_size=header_font_size,
                         layers_clusters_dict=layers_clusters_dict,
                         heat_map_connections=add_heat_map_on_connections,
                         imgs_path=img_clusters_dir,
                         saving_dir=saving_dir,
                         heat_map_path=heat_map_dir,
                         image_inference=True)
