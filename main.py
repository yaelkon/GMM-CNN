import os
import numpy as np
import time
import tensorflow as tf

from keras.applications.vgg16 import preprocess_input
from keras.utils import multi_gpu_model
from utils.file import save_to_file, load_from_file
from utils.imagenet_utils import ImageNet_Generator, imagenet_preprocessing
from utils.getters import *
from utils.gmm_utils import *

""" 
An experiment pip for building a Deep GMM with task transfer from a pre-trained network(ResNet50).
Define your experiment parameters below"""

# Insert experiment parameters
FILES_DIR = 'D:/Yael/ILSVRC2012'
TRAIN_DIR = 'D:/Yael/ILSVRC2012/all_train_images/'
VAL_DIR = 'D:/Yael/ILSVRC2012/val/'
SAVING_DIR = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16'
WEIGHTS_DIR = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block1_conv1/' \
               'gmm_model_with_classifier_gradient_K=60_squirrel_monkey_20191128_174617/weights.08.hdf5'
# Loading the data
y_train = np.load(os.path.join(FILES_DIR, 'train_labels_one_hot_shuffled.npy'))
X_train_filenames = np.load(os.path.join(FILES_DIR, 'train_filenames_shuffled.npy'))
y_val = np.load(os.path.join(FILES_DIR, 'val_y_one_hot.npy'))
X_val_filenames = np.load(os.path.join(FILES_DIR, 'val_filenames.npy'))

# --------- GMM parameters
gpu_model = False
gpu_list = None
if gpu_list is None:
    n_gpu = 0
else:
    n_gpu = len(gpu_list)
load_prev_weights_from_file = False
pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block1_conv1/' \
               'gmm_model_with_classifier_gradient_K=60_squirrel_monkey_20191128_174617/weights.08.hdf5'
init_gmm_layer_according_to_conv_gmm_params = True  # If true - the gmm mean and std will be initialized according to
                                                    # the mean and std of the test set.
init_params_mode = 1
optimize_gmm_with_classifier = True
gmm_training = 'with_classifier_gradient' # 'with_llr_gradient', 'with_classifier_gradient'
to_train_gmm_std = True # True - gmm's std will be trainable parameters
                        # False - gmm's std will be equal to the identity matrix


# --------- Data parameters
data_name = 'imagenet'
data_augmentation = False
subtract_pixel_mean = []
n_classes = 1000
# Define model
# --------- Model parameters
# Network parameters:
depth = 50  # the number of conv2D layers the model will contained.
input_shape = (224, 224, 3)
padding = 'same'
batch_normalization = False
set_classification_as_output = False
# Training parameters:
es_patience = 8    # Number of patience epochs for EarlyStopping.
batch_size = 15
num_epochs = 1 #8
validation_split = []
optimizer = 'adam'
lr = 0.0
network_type = 'resnet50'   # options: 'resnet50'- a well trained network on Imagenet dataset,
                         #  'resnet_for_cifar10'- a resnet network build from scratch,
                         #  'simple'- a small network mostly for debugging use

train_on_classes = [682, 948, 953]
main_class = 953
main_class_name = 'pineapple'
if gmm_training:
    freeze_regular_layers = True
    data_augmentation = False

else:
    gmm_layers = None
    data_augmentation = True
    freeze_regular_layers = False
    weights_dir = None
    init_gmm_layer_according_to_conv_gmm_params = False
if load_prev_weights_from_file:
    init_gmm_layer_according_to_conv_gmm_params = False
    init_params_mode = None

# ------------------    Training begin  -------------------------------------
Params = GetDefaultParameters()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Insert the user input to Params
Params['Data']['name'] = data_name
Params['Data']['n_classes'] = n_classes
Params['Data']['data_augmentation'] = data_augmentation
Params['Data']['subtract_pixel_mean'] = subtract_pixel_mean
Params['Data']['train_on_classes'] = train_on_classes
Params['Data']['major_class'] = main_class
Params['Data']['major_class_name'] = main_class_name

Params['Train']['batch_size'] = batch_size
Params['Train']['n_epochs'] = num_epochs
Params['Train']['optimizer'] = optimizer
Params['Train']['es_patience'] = es_patience
Params['Train']['validation_spilt_fraction'] = validation_split
Params['Train']['lr'] = lr

Params['GMM']['init_with_respect_to_the_data'] = init_gmm_layer_according_to_conv_gmm_params
Params['GMM']['init_params_mode'] = init_params_mode
Params['GMM']['optimize_with_classifier_loss'] = optimize_gmm_with_classifier
Params['GMM']['training_type'] = gmm_training
Params['GMM']['train_std'] = to_train_gmm_std

Params['Network']['name'] = network_type
Params['Network']['input_shape'] = input_shape
Params['Network']['padding'] = padding
Params['Network']['batch_normalization'] = batch_normalization
Params['Network']['freez_layers'] = freeze_regular_layers
Params['Network']['weights_dir'] = []
Params['Network']['gpu_model'] = gpu_model
Params['Network']['set_classification_as_output'] = set_classification_as_output

if network_type == 'resnet50':
    depth = 50
Params['Network']['depth'] = depth

# Filtering out the un-use data
if len(train_on_classes) > 0:
    train_on_classes = np.asarray(train_on_classes)
    #   Update training data
    relevant_train_inds = np.where(y_train[:, train_on_classes] == 1)[0]
    X_train_filenames = X_train_filenames[relevant_train_inds]
    y_train = y_train[relevant_train_inds, :]
    #   Update validation data
    relevant_val_inds = np.where(y_val[:, train_on_classes] == 1)[0]
    X_val_filenames = X_val_filenames[relevant_val_inds]
    y_val = y_val[relevant_val_inds, :]

    print(f'Training for main class: {main_class_name}')

if to_train_gmm_std:
    err_json_name = gmm_training
else:
    err_json_name = gmm_training + '_without_std'

layers_to_model_vec = ['classification']
# 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'fc1'
# 'add_2', 'add_4', 'add_6', 'add_8', 'add_10', 'add_12', 'add_14', 'add_16'
gaussians_vec = [1000]

for layer_to_model in layers_to_model_vec:

    gmm_errors_dict = {}
    Params['GMM']['layers_to_model'] = [layer_to_model]
    prev_error = 1
    reached_satisfy_improvement = False
    layer_saving_path = os.path.join(SAVING_DIR, layer_to_model)

    if os.path.isfile(os.path.join(layer_saving_path, 'gmm_errors_'+err_json_name+'.json'))\
            and os.access(os.path.join(layer_saving_path, 'gmm_errors_'+err_json_name+'.json'), os.R_OK):
        # checks if file exists
        print("File exists and is readable")
        print("Load results...")
        gmm_errors_dict = load_from_file(layer_saving_path, ['gmm_errors_'+err_json_name])[0]

    for n_gaussians in gaussians_vec:
        if layer_to_model == 'block2_conv1':
            n_gaussians = 100
            pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block2_conv1/' \
                            'gmm_model_with_classifier_gradient_K=100_squirrel_monkey_20191128_201908/weights.06.hdf5'
        elif layer_to_model == 'block3_conv1':
            n_gaussians = 200
            pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block3_conv1/' \
                            'gmm_model_with_classifier_gradient_K=200_pineapple_beforeRelu_20191203_004112/weights.06.hdf5'
        elif layer_to_model == 'block4_conv1':
            n_gaussians = 450
            pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block4_conv1/' \
                            'gmm_model_with_classifier_gradient_K=450_pineapple_beforeRelu/weights.04.hdf5'
        elif layer_to_model == 'block5_conv1':
            n_gaussians = 1500
            pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/block5_conv1/' \
                            'gmm_model_with_classifier_gradient_K=1500_pineapple_beforeRelu/weights.03.hdf5'
        elif layer_to_model == 'fc1':
            n_gaussians = 100
            pweights_dir = 'G:/My Drive/StoragePath/ExpResults/Yael/vgg16/fc1/' \
                           'gmm_model_with_classifier_gradient_K=100_pineapple_beforeRelu_20191204_234654/weights.04.hdf5'

        if not reached_satisfy_improvement:
            if gpu_model:
                batch_size = 1 * len(gpu_list)
            else:
                batch_size = 1
            Params['exp_path'] = os.path.join(*[SAVING_DIR, layer_to_model,
                                                'gmm_model_' + err_json_name + '_K=' + str(n_gaussians)
                                                + '_' + main_class_name + '_' + time.strftime('%Y%m%d_%H%M%S')])
            Params['Train']['batch_size'] = batch_size
            Params['GMM']['n_gaussians'] = n_gaussians
            print('optimize layer: ', layer_to_model)
            print('number of gaussians in the experiment: ', n_gaussians)
            print('batch_size: ', batch_size)

            if gpu_model:
                with tf.device('/cpu:0'):
                    model = GetModel(Params)

                origin_keras_model = model.keras_model
                model.keras_model = multi_gpu_model(model.keras_model, gpus=n_gpu)
                model.compile_model()

            else:
                model = GetModel(Params)
                model.compile_model()

            Params['Train']['val_indices'] = 'D:/Yael/ILSVRC2012/val_filenames.npy'
            Params['Train']['val_labels'] = 'D:/Yael/ILSVRC2012/val_y_keras_one_hot.npy'

            # Prepering outputs for gmm and classification model
            if Params['GMM']['layers_to_model'] is not None:
                num_gmm_outputs = len(model._output_layers)
                train_labels = []
                val_labels = []
                for i in range(num_gmm_outputs):
                    train_labels.append(y_train)
                    val_labels.append(y_val)
            else:
                train_labels = y_train
                val_labels = y_val

            # results = origin_keras_model.evaluate(x=validation_batch_generator, batch_size=5)
            if load_prev_weights_from_file:
                print('Load weights from previous experiment')
                origin_keras_model.load_weights(pweights_dir, by_name=True)

            elif init_gmm_layer_according_to_conv_gmm_params:
                print('Initialising weights')
                if layer_to_model == 'classification':
                    data = None
                else:
                    data = imagenet_preprocessing(X_train_filenames[:500], train_dir)
                    data = preprocess_input(data)

                if gpu_model:
                    keras_model = model.keras_model.get_layer(index=1+len(gpu_list))
                    layers_gmm_params = calc_layer_mean_and_std(model, data, keras_model=keras_model, mode=2)
                    model.keras_model.layers[1+len(gpu_list)] = set_gmm_weights(keras_model, layers_gmm_params,
                                                                                set_std=to_train_gmm_std)
                else:
                    layers_gmm_params = calc_layer_mean_and_std(model, data, keras_model=model.keras_model, mode=2)
                    model.keras_model = set_gmm_weights(model.keras_model, layers_gmm_params,
                                                                                set_std=to_train_gmm_std)
            save_to_file(file_dir=Params['exp_path'],
                         objs_name=['params'],
                         objs=[Params])

            training_batch_generator = ImageNet_Generator(image_filenames=X_train_filenames, labels=train_labels, batch_size=batch_size, images_dir=train_dir)
            validation_batch_generator = ImageNet_Generator(image_filenames=X_val_filenames, labels=val_labels, batch_size=batch_size, images_dir=val_dir)
            history = model.fit_generator(datagen=training_batch_generator,
                                          epochs=Params['Train']['n_epochs'],
                                          batch_size=batch_size,
                                          validation_data=validation_batch_generator,
                                          gpus=gpu_list)

            # saving results
            Params['Summary']['val_loss'] = round(history.history.get('val_loss')[-1], 4)
            gmm_classifier_name = 'val_' + model.classifiers_dict[model.gmm_dict[layer_to_model]] + '_acc'
            Params['Summary']['val_gmm_classifier_acc'] = round((history.history.get('val_acc')[-1]), 3)

            save_to_file(file_dir=Params['exp_path'],
                         objs_name=['params'],
                         objs=[Params])

            # cheking whether the gmm classifier error improved
            curr_err = Params['Summary']['val_gmm_classifier_acc']
            gmm_errors_dict[n_gaussians] = curr_err

        save_to_file(file_dir=layer_saving_path,
                     objs_name=['gmm_errors_'+err_json_name],
                     objs=[gmm_errors_dict])

