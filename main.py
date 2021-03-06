import os
import numpy as np
from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from keras.datasets import cifar10
from keras.utils import to_categorical

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
""" 
An experiment pip for modeling CNN layers with GMM.
Define your experiment parameters below
"""

SAVING_DIR = pjoin(*['C:', os.environ["HOMEPATH"], 'Desktop', 'tmp', 'resnet20_cifar10'])

# A pre-trained network's weights - optional
WEIGHTS_DIR = pjoin(os.path.abspath(os.getcwd()), 'utils', 'cifar10_resnet20_weights.97.hdf5')

# --------- GMM parameters
# Choose between 'generative' or 'discriminative' training loss
GMM_training_method = 'discriminative'

# --------- Data parameters
input_shape = (32, 32, 3)

# --------- Model parameters
network_name = 'resnet20'
# Specify the layer name as str to model or a list contains str layer names for multiple modeled layers
layer_to_model = ['add_2', 'add_4', 'add_6', 'add_8', 'classification']
# Specify the number of clusters each GMM will contain.
# The clusters order has to be matched to the order specified in 'layer_to_model' arg.
n_gaussians = [500, 500, 500, 500, 10]

# --------- Training parameters
batch_size = 5
num_epochs = 10


# -----------------------   Prepare cifar 10 dataset    --------------------------

(x_train, y_train), (x_val, y_val) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# Normalize the data
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

# subtract pixel mean
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_val -= x_train_mean

# ------------------------   Begin Training  -------------------------------------

model = GMM_CNN( n_gaussians=n_gaussians,
                 input_shape=input_shape,
                 n_classes=10,
                 training_method=GMM_training_method,
                 saving_dir=SAVING_DIR,
                 layers_to_model=layer_to_model,
                 network_name=network_name,
                 set_classification_layer_as_output=True,
                 weights_dir=WEIGHTS_DIR )

model.build_model()
model.compile_model()

print('Initialising GMM parameters')

layers_gmm_params = model.calc_modeled_layers_mean_and_std(x_train[:1000])
model.set_weights(layers_gmm_params)

# Fit the labels size according to the number of outputs layers
n_outputs = len(model.output_layers)
train_labels = []
val_labels = []
for i in range(n_outputs):
    train_labels.append(y_train)
    val_labels.append(y_val)

print('Optimizing GMM params for layer: ', layer_to_model)
print('Number of gaussians in the experiment: ', n_gaussians)
print('Batch size: ', batch_size)

history = model.fit(x=x_train, y=train_labels,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_data=(x_val, val_labels))

