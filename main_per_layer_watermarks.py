import os
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import join as pjoin
from gmm_cnn import GMM_CNN
from keras.datasets import cifar10
from keras.utils import to_categorical
from utils.vis_utils import get_cifar10_labels, get_cifar10_watermarks_dict
from utils.data_preprocessing import add_watermark_by_class
from utils.file import load_data_from_file

parser = argparse.ArgumentParser(description='Plane experiment')

parser.add_argument('--cls1', dest='cls1',
                    help='The name of the first class to add watermarks on top of its images',
                    required=False, type=str)
parser.add_argument('--cls2', dest='cls2',
                    help='The name of the second class to add watermarks on top of its images',
                    required=False, type=str)

args = parser.parse_args()

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
""" 
An experiment pip for modeling CNN layers with GMM.
Define your experiment parameters below
"""

# , time.strftime('%Y%m%d_%H%M%S')
# A pre-trained network's weights - optional
UTILS_DIR = pjoin(os.path.abspath(os.getcwd()), 'utils')
WEIGHTS_DIR = pjoin(UTILS_DIR, 'cifar10vgg.h5')

try:
    FONT_DIR = pjoin(UTILS_DIR, 'Arialn.ttf')

except:

    FONT_DIR = None

IS_WATERMARK_EXP = True
# --------- GMM parameters
# Choose between 'generative' or 'discriminative' training loss
GMM_training_method = None

# --------- Data parameters
input_shape = (32, 32, 3)

# --------- Model parameters
network_name = 'vgg16'

SAVING_DIR = 'G:\\My Drive\\Research\\My Papers\\TVCG paper\\experiments\\tmp'
# Specify the layer name as str to model or a list contains str layer names for multiple modeled layers
layer_to_model = None
# layer_to_model = ['conv2d_8', 'conv2d_11', 'classification']
# Specify the number of clusters each GMM will contain.
# The clusters order has to be matched to the order specified in 'layer_to_model' arg.
n_gaussians = []

# --------- Training parameters
batch_size = 80
num_epochs = 250
add_top = False
max_channel_clustering = False
# -----------------------   Prepare cifar 10 dataset    --------------------------
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
# x_train = x_train[:100]
# y_train = y_train[:100]
# x_val = x_val[:100]
# y_val = y_val[:100]
labels = get_cifar10_labels()

if IS_WATERMARK_EXP:
    if not (hasattr(args, 'cls1') and hasattr(args, 'cls2')):
        raise AttributeError (f'Watermark experiment set to {IS_WATERMARK_EXP}, but there are not classes names specified in argparse for cls1 and cls2')
    SAVING_DIR = pjoin(SAVING_DIR, 'Watermark')
    DATA_DIR = pjoin(SAVING_DIR, 'Watermark_Data')
    SAVING_DIR = pjoin(*[SAVING_DIR, network_name, args.cls1 + '_' + args.cls2])
    base_watermarks_dict = get_cifar10_watermarks_dict()
    WM_dict = {args.cls1: base_watermarks_dict[args.cls1], args.cls2: base_watermarks_dict[args.cls2]}

    print('Preparing train watermark dataset')
    x_train = add_watermark_by_class(WM_dict,
                                     x_train,
                                     y_train,
                                     labels,
                                     train0validation1=0,
                                     save_dataset=True,
                                     saveing_dir=DATA_DIR,
                                     fonttype=FONT_DIR if FONT_DIR is not None else 'Ubuntu-R.ttf')
    print('Preparing validation watermark dataset')
    x_val = add_watermark_by_class(WM_dict,
                                   x_val,
                                   y_val,
                                   labels,
                                   train0validation1=1,
                                   save_dataset=True,
                                   saveing_dir=DATA_DIR,
                                   fonttype=FONT_DIR if FONT_DIR is not None else 'Ubuntu-R.ttf')

# Convert class vectors to binary class matrices.

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
# Normalize the data
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

if network_name == 'vgg16':
    # these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    x_train = (x_train - mean) / (std + 1e-7)
    x_val = (x_val - mean) / (std + 1e-7)

else:
    x_train /= 255
    x_val /= 255

    # subtract pixel mean
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_val -= x_train_mean

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# ------------------------   Begin Training  -------------------------------------
# for i in range(len(layer_to_model)):
#     n_g = n_gaussians[i]
#     layer = layer_to_model[i]
# for n_g, layer in zip(n_gaussians, layer_to_model):
EXP_DIR = pjoin(*[SAVING_DIR, 'baseline'])
model = GMM_CNN( n_gaussians=n_gaussians,
                 input_shape=input_shape,
                 n_classes=10,
                 training_method=GMM_training_method,
                 saving_dir=EXP_DIR,
                 layers_to_model=layer_to_model,
                 network_name=network_name,
                 set_classification_layer_as_output=True,
                 weights_dir=WEIGHTS_DIR,
                 freeze=False,
                 add_top=add_top,
                 max_channel_clustering=max_channel_clustering,
                 batch_size=batch_size
                 )

model.build_model()
model.compile_model()

print('Initialising GMM parameters')

# layers_gmm_params = model.calc_modeled_layers_mean_and_std(x_train[:2])
# model.set_weights(layers_gmm_params)

# Fit the labels size according to the number of outputs layers
n_outputs = len(model.output_layers)
train_labels = []
val_labels = []
if n_outputs == 1:
    train_labels = y_train
    val_labels = y_val
else:
    for i in range(n_outputs):
        train_labels.append(y_train)
        val_labels.append(y_val)

print('Optimizing GMM params for layer: ', layer_to_model)
print('Number of gaussians in the experiment: ', n_gaussians)
print('Batch size: ', batch_size)
history = model.fit_generator(datagen=datagen, x=x_train, y=train_labels,
                              batch_size=batch_size,
                              epochs=num_epochs,
                              validation_data=(x_val, val_labels))

