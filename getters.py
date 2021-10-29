import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10, cifar100
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from gmm_cnn import GMM_CNN
# from Sandboxes.Yael.callbacks import lr_schedule


def GetDefaultParameters():
    Params = {}

    Params['Seed'] = 101

    Params['Train'] = {}
    Params['Train']['batch_size'] = 10
    Params['Train']['n_epochs'] = 50
    Params['Train']['AUGMENT'] = False

    Params['Data'] = {}
    Params['Data']['validation_split'] = 0.2
    Params['Data']['name'] = 'Cifar10'
    Params['Data']['n_classes'] = 10
    Params['Data']['data_augmentation'] = False
    Params['Data']['subtract_pixel_mean'] = False

    Params['GMM'] = {}
    Params['GMM']['n_gaussians'] = 30

    Params['Network'] = {}
    Params['Network']['name'] = 'resnet_for_cifar10'
    Params['Network']['gmm_layers'] = None

    Params['Summary'] = {}

    return Params

def GetData(Params):
    NUM_CLASSES = Params['Data']['n_classes']

    if Params['Data']['name'] == 'Cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Wrong tag
        y_test[2405] = 6
    elif Params['Data']['name'] == 'Cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # Normalize the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # If subtract pixel mean is enabled
    if Params['Data']['subtract_pixel_mean']:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Run training, with or without data augmentation.
    if not Params['Data']['data_augmentation']:
        print('Not using data augmentation.')

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:

        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    Data ={}
    Data['x_Train'] = x_train
    Data['y_Train'] = y_train
    Data['x_Test'] = x_test
    Data['y_Test'] = y_test

    if not Params['Data']['data_augmentation']:
        return Data

    else:
        return Data, datagen

def GetModel(Params):

    if Params['optimizer'] == 'SGD':
        optimizer = SGD(lr=0.01, decay=5e-4, momentum=0.9)  # it is possible to change the optimizer but not the lr field
    elif Params['Train']['optimizer'] == 'Adam':
        optimizer = Adam(lr=0.01, decay=5e-4)

    n_gaussians = Params['n_gaussians']
    gmm_layers = Params['modeled_layers']
    if 'training_method' in Params:
        gmm_training_type = Params['training_method']
    else:
        gmm_training_type = 'discriminative'

    if 'set_gmm_activation_layer_as_output' in Params:
        set_gmm_activation_layer_as_output = Params['set_gmm_activation_layer_as_output']
    else:
        set_gmm_activation_layer_as_output = False
    if 'set_gmm_layer_as_output' in Params:
        set_gmm_layer_as_output = Params['set_gmm_layer_as_output']
    else:
        set_gmm_layer_as_output = False

    if 'set_classification_as_output' in Params:
        set_classification_as_output = Params['set_classification_as_output']
    else:
        set_classification_as_output = True

    input_shape = Params['input_shape']
    network_name = Params['network_name']
    freeze = True
    weights_dir = Params['weights_dir']

    n_classes = Params['n_classes']

    model_path = Params['model_path']

    model = GMM_CNN( n_gaussians=n_gaussians,
                       input_shape=input_shape,
                       n_classes=n_classes,
                       optimizer=optimizer,
                       saving_dir=model_path,
                       layers_to_model=gmm_layers,
                       training_method=gmm_training_type,
                       set_gmm_activation_layer_as_output=set_gmm_activation_layer_as_output,
                       set_gmm_layer_as_output=set_gmm_layer_as_output,
                       set_classification_layer_as_output=set_classification_as_output,
                       network_name=network_name,
                       seed=Params['seed'],
                       freeze=freeze,
                       weights_dir=weights_dir
                     )

    model.build_model()
    return model

def filter_data_by_class(class_name, data, labels, labels_name):

    class_index = labels_name.index(class_name)
    data_indices = np.where(labels[:, class_index] == 1)

    filtered_data = data[data_indices]
    filtered_labels = labels[data_indices]

    return filtered_data, filtered_labels

def load_model_from_file(path):
    """Load the model of the model from a file.
    Args:
        path (str): Path to the weights file.
    """
    model = load_model(path)
    print('Loaded model from file.')

    return model