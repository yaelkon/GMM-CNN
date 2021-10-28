import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import copy
import time
from utils.gmm_utils import get_gmm_weights_dict, set_gmm_weights
from utils.file import load_from_file, save_to_file
from utils.getters import GetModel
from utils.file import makedir

# Parameters

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
modeled_layers_name = ['conv2d_1', 'conv2d_3',
                       'conv2d_5', 'conv2d_8',
                       'conv2d_11', 'classification']
optimal_clusters_vec = [50, 50, 50, 50, 50, 10]

main_root = os.path.join(*['G:', 'My Drive', 'Research', 'My Papers', 'TVCG paper', 'experiments', 'Watermarks', 'vgg16', 'horse_ship'])
merged_path = os.path.join(main_root, 'merged_path')
merged_weights_path = os.path.join(merged_path, 'weights.00.hdf5')
makedir(merged_path)
merged_params = None
gmm_weights_dict = {}

# ---------------------------------------------------------------------
for modeled_layer, optimal_k in zip(modeled_layers_name, optimal_clusters_vec):
    print(f'Gathering {modeled_layer} weights')
    model_dir = os.path.join(*[main_root, modeled_layer])
    #     Load experiment params and model
    params = load_from_file(model_dir, ['config'])[0]
    params['model_path'] = model_dir
    params['weights_dir'] = os.path.join(*[main_root, 'baseline', 'weights.225.hdf5'])
    # params['n_gaussians'] = 2
    #     Find the most updated weights file
    list_of_weights = glob.glob(os.path.join(model_dir, 'weights.*.hdf5'))
    weights_dir = list_of_weights[-1]

    if merged_params is None:
        merged_params = copy.deepcopy(params)
        merged_params['n_gaussians'] = optimal_clusters_vec
        merged_params['model_path'] = merged_path
        merged_params['set_classification_as_output'] = True
    else:
        merged_params['modeled_layers'] = merged_params['modeled_layers'] + params['modeled_layers']
        # merged_params['Summary']['val_gmm_classifier_acc'] = merged_params['Summary']['val_gmm_classifier_acc'] + \
        #                                                      params['Summary']['val_gmm_classifier_acc']

    model = GetModel(params)
    model.load_weights_from_file(weights_dir)

    gmm_weights_dict_temp = get_gmm_weights_dict(model)

    for gmm_layer_name in gmm_weights_dict_temp:
        if gmm_layer_name not in gmm_weights_dict:
            gmm_weights_dict.update({gmm_layer_name: gmm_weights_dict_temp[gmm_layer_name]})
print('Prepering merged model')
merged_model = GetModel(merged_params)
merged_model.compile_model()
merged_model.keras_model = set_gmm_weights(merged_model.keras_model, gmm_weights_dict, initialization=False)
merged_model.save_model_plot()
merged_model.save_weights_to_file(merged_weights_path)
save_to_file(merged_path, ['config'], [merged_params])
