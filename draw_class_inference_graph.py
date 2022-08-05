import glob
import scipy.io as sio
import os

from keras.datasets import cifar10
from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from utils.file import load_from_file, makedir
from utils.gmm_utils import *
from utils.vis_utils import get_cifar10_labels
from layers import get_layer_output
from inference_graph import Inference_Graph
from utils.IG_utils import gather_coOccurrance_mat, get_IG_connections_dict_LR,\
     create_connections_heat_maps
from receptivefield import ReceptiveField


EXP_PATH = pjoin(*['C:', os.environ["HOMEPATH"], 'Desktop', 'tmp', 'resnet20_cifar10'])
head_classes = ['truck']
n_nodes_in_graph = 3
edge_label_font_size = '22'
header_font_size = '40'
add_heat_map_on_connections = False
calc_heat_map_on_connections = False
show_head_class = True
to_color_edges = True
add_quantities_on_edges = True

# Specify the layers to visualize and the visualization technique for each layer ('rectangle'/'patches')
vis_option = {'add_2': 'rectangle',
              'add_4': 'rectangle',
              'add_6': 'rectangle',
              'add_8': 'rectangle'
              }

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

layers_to_inference = []
for key in vis_option:
    layers_to_inference.append(key)
layers_to_inference.append(model.keras_model.output_names[0])
# make clusters saving directory
vis_dir = pjoin(EXP_PATH, 'clusters_vis')

# get data
(_, _), (x_val, y_val) = cifar10.load_data()

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
saving_dir = pjoin(*[results_dir, 'class_tree'])
makedir(saving_dir)

RF = ReceptiveField(model.keras_model)

layers_clusters_dict = {}
appearance_hist_dict = {}

for layer_name in layers_to_inference:
    gmm_name = model.get_correspond_gmm_layer( layer_name )
    n_clusters = model.keras_model.get_layer( gmm_name ).n_clusters
    layers_clusters_dict[layer_name] = n_clusters

# Gather co-occurrence matrices
global_appearance_hist_dict = {}
calc_global_flag = False
explained_layer = layers_to_inference[-1]

# Load the appearance matrices
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
                                                          num_of_iterations=5,
                                                          saving_flag=True,
                                                          return_flag=True)

test_pred = get_layer_output(keras_model=model.keras_model, layer_name=layers_to_inference[-1], input_data=x_val)
preds_ind = np.argmax(test_pred, axis=1)

# Start making inference tree for every class
for head_class in head_classes:

    class_index = classes_labels.index(head_class)
    IT_name = 'inference_' + head_class + '_graph'

    print(f'collecting connections for class \'{head_class}\'')

    imgs_class_inds = np.where(preds_ind == class_index)[0]
    imgs = x_val[imgs_class_inds]
    local_appearance_hist_dict = gather_coOccurrance_mat(gmm_model=model,
                                                         exp_path=EXP_PATH,
                                                         RF=RF,
                                                         layers_to_inference=layers_to_inference,
                                                         data=imgs,
                                                         num_of_iterations=1,
                                                         saving_flag=False,
                                                         return_flag=True)
    connections_dict = get_IG_connections_dict_LR(class_index=class_index,
                                                  global_appearance_hist_dict=global_appearance_hist_dict,
                                                  local_appearance_hist_dict=local_appearance_hist_dict,
                                                  layers_to_inference=layers_to_inference,
                                                  n_nodes=n_nodes_in_graph)
    if calc_heat_map_on_connections:
        create_connections_heat_maps(gmm_model=model,
                                     exp_path=EXP_PATH,
                                     RF=RF,
                                     data=x_val,
                                     connections_dict=connections_dict,
                                     saving_dir=heat_map_dir)

    IT = Inference_Graph(name=IT_name,
                         head_class_name=head_class,
                         head_class_index=class_index,
                         connections_dict=connections_dict,
                         node_type=vis_option,
                         n_nodes=n_nodes_in_graph,
                         edge_label_font_size=edge_label_font_size,
                         header_font_size=header_font_size,
                         layers_clusters_dict=layers_clusters_dict,
                         heat_map_connections=add_heat_map_on_connections,
                         imgs_path=clusters_dir,
                         saving_dir=saving_dir,
                         heat_map_path=heat_map_dir,
                         show_head_class=show_head_class,
                         to_color_edges=to_color_edges,
                         add_quantities_on_edges=add_quantities_on_edges,
                         )

