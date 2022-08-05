import os
import numpy as np

from sklearn.metrics import confusion_matrix
from keras.layers import (Input, Dense, GlobalAveragePooling2D, Flatten, Lambda, Activation)
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.losses import categorical_crossentropy
from keras.utils import plot_model, CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from losses import build_gmm_loss
from layers import GMM, gmm_bayes_activation, get_layer_output
from utils.file import makedir, save_to_file
from utils.resnet import build_resnet


class Encoder( object ):
    """Auxiliary class to define encoding layers common to GaussianMixture and CNN.

    Args:


    """

    def __init__(self, num_classes, freeze, weights_dir, weight_decay, add_top):
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.freeze = freeze
        self.weights_dir = weights_dir
        self.add_top = add_top

    def _build_encoder_layers(self, input_layer, name, gmm_layers=None):
        """Stacks sequence of conv/pool layers to make the encoder half.
        """
        encoded = []

        if name == 'resnet50':
            base_model = ResNet50( weights='imagenet', include_top=(not self.add_top), input_tensor=input_layer )
        elif name == 'resnet20':
            base_model = build_resnet( depth=20, input_layer=input_layer, n_classes=self.num_classes )
        elif name == 'vgg16':
            base_model = VGG16( weights='imagenet', include_top=(not self.add_top), input_tensor=input_layer )
        else:
            raise TypeError( 'The model type cane be either resnet20, resnet50 or vgg16' )

        if self.weights_dir is not None and len( self.weights_dir ) > 0:
            base_model.load_weights( self.weights_dir )
            print( 'Loaded weights from file.' )
        # Freeze layers
        if self.freeze:
            print( 'freezing networks weights' )
            for layer in base_model.layers:
                layer.trainable = False

        layer = base_model.layers[-1].output
        encoded.append( layer )

        if gmm_layers:
            if gmm_layers == 'all':
                for l in base_model.layers:
                    if type( l ).__name__ == 'Conv2D':
                        layer = l.output
                        encoded.append( layer )
            else:
                for l in gmm_layers:
                    layer = base_model.get_layer( name=l ).output
                    encoded.append( layer )
        return encoded


class GMM_CNN( Encoder ):
    """Builds and compiles a model for representation learning GMM:
        The network structure is as follows:
        Input -> Encoder -> Output
        The gmm layers output representing P(x,h=k) for spatial example x and Gaussian Mixture Density k.

    Args:
        n_gaussians (list): Number of Gaussians in the mixture for each modeled layer.
        input_shape (tuple): Shape of input 3D array.
        optimizer (str): Name of Keras optimizer. An optimizer object can also be passed directly.
        model_path (str): Path to the location where the model files will be saved.
        weights_name_format(str): Format of weight checkpoint files (refer to the Keras documentation for details).
        histogram_freq (int): Frequency (in epochs) for saving weights, activations and gradients
        histograms in Tensorboard.If 0, no histograms will be computed.
        Other arguments passed to Encoder instances.

    Methods:
        build_model: Builds and compiles the model.
        summary: Print summary of model to stdout
        load_weights_from_file: Load the weights of the model from a file.
        fit_generator: Train the model on data yielded batch-by-batch by a generator.
    """

    def __init__(self,
                 n_gaussians=500,
                 input_shape=(224, 224, 3),
                 n_classes=1000,
                 weight_decay=l2( 1e-4 ),
                 seed=101,
                 hyper_lambda=1,
                 optimizer='Adam',
                 saving_dir='/tmp/',
                 weights_name_format='weights.{epoch:02d}.hdf5',
                 training_method='generative',
                 network_name='vgg16',
                 set_gmm_activation_layer_as_output=False,
                 set_gmm_layer_as_output=False,
                 set_classification_layer_as_output=True,
                 freeze=True,
                 add_top=False,
                 layers_to_model=None,
                 weights_dir=None):

        # Fitting n_gaussians vector to be as the same length as the modeled layers
        if isinstance( layers_to_model, str ):
            layers_to_model = [layers_to_model]
        if isinstance( layers_to_model, list ):
            if isinstance( n_gaussians, int ):
                n_gaussians = [n_gaussians]
            if len( n_gaussians ) != len( layers_to_model ):
                raise ValueError( '\'n_gaussians\' size have to by the same size and order as \'gmm_layers\'' )
        elif layers_to_model is not None:
            raise ValueError( 'GMM_layers have to receive a list (for multiple) or a string (for a single) value'
                              ' contains the layers name you want to model' )

        if training_method != 'discriminative' and training_method != 'generative':
            raise ValueError( 'GMM training method must be either generative or discriminative' )

        super( GMM_CNN, self ).__init__( n_classes, freeze, weights_dir, weight_decay, add_top )

        self.n_gaussians = n_gaussians
        self.model_path = saving_dir
        self.weights_name_format = weights_name_format
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.modeled_layers = layers_to_model
        self.training_method = training_method
        self.set_gmm_layer_as_output = set_gmm_layer_as_output
        self.set_gmm_activation_layer_as_output = set_gmm_activation_layer_as_output
        self.set_classification_layer_as_output = set_classification_layer_as_output
        self.network_name = network_name
        self.seed = seed
        self.hyper_lambda = hyper_lambda
        self.freeze = freeze

        self.input_layer = None
        self.output_layers = None
        self.network_output_layer_name = None
        self.losses = None
        self.metrics_dict = None
        self.gmm_dict = {}
        self.gmm_activations_dict = {}
        self.gmm_layers = []
        self.classifiers_dict = {}

    def save_model_plot(self):
        plot_model( self.keras_model, to_file=os.path.join( self.model_path, 'model.png' ) )

    def _build_callbacks(self):
        """Builds callbacks for training model.
        """

        if self.training_method == 'discriminative':
            monitoring_name = 'val_' + self.classifiers_dict[self.gmm_dict[self.modeled_layers[0]]] + '_acc'
        else:
            monitoring_name = 'val_' + self.gmm_layers[0] + '_loss'

        weights_checkpointer = ModelCheckpoint( filepath=os.path.join( self.model_path, self.weights_name_format ),
                                                monitor=monitoring_name, save_best_only=True, save_weights_only=False )
        self.save_model_plot()
        self.save_config()

        epoch_logger = CSVLogger( os.path.join( self.model_path, 'epoch_log.csv' ) )
        tensorboard_path = os.path.join( self.model_path, 'tensorboard' )
        tensorboard = TensorBoard( log_dir=tensorboard_path, update_freq='epoch' )

        terminator = TerminateOnNaN()

        return [weights_checkpointer, epoch_logger, tensorboard, terminator]

    def _build_classifier_layers(self, encoded):
        """Builds layers for classification on top of encoder layers.
        """
        h = Flatten()( encoded )
        # Add classifier on top
        y = Dense( self.n_classes, activation='softmax', name='classification' )( h )

        return y

    def _build_gmm_classifier_layers(self, encoded=None):
        """Builds the output layer that will optimize GMM parameters with a classifier alongside
        with log likelihood gmm function.
        """
        outputs_array = []
        for i in range( len( encoded ) ):
            x = encoded[i]
            layer_name = self.get_correspond_gmm_layer( self.modeled_layers[i] )

            if self.training_method == 'generative':
                x = Lambda( lambda x: K.stop_gradient( x ), name='stop_grad_layer_classifier_%d' % (i + 1) )( x )

            if len( x.shape ) == 4:
                x = GlobalAveragePooling2D( name='GAP_' + layer_name )( x )
            x = Dense( self.n_classes, activation='softmax', name='classifier_' + layer_name )( x )
            outputs_array.append( x )
            self.classifiers_dict.update( {layer_name: 'classifier_' + layer_name} )

        return outputs_array

    def _build_gmm_layers(self, encoded=None):
        """Builds the output layer that parametrizes a Gaussian Mixture Density.
        """
        outputs_array = []
        layers_array = []
        for i in range( len( encoded ) ):
            layer_name = encoded[i].name.rsplit( '/', 1 )[0]
            x = encoded[i]

            if len( x.shape ) == 2:
                gmm_name = 'gmm_' + layer_name
                stop_grad_layer = Lambda( lambda x: K.stop_gradient( x ), name='stop_grad_layer_'
                                                                               + layer_name )( x )
                layer = GMM( n_clusters=self.n_gaussians[i],
                             seed=self.seed,
                             name=gmm_name,
                             modeled_layer_type='Dense' )( stop_grad_layer )
                self.gmm_dict.update( {layer_name: gmm_name} )
                self.gmm_layers.append( gmm_name )

            elif len( x.shape ) == 4:
                gmm_name = 'gmm_' + layer_name
                stop_grad_layer = Lambda( lambda x: K.stop_gradient( x ), name='stop_grad_layer_' + layer_name )( x )
                layer = GMM( n_clusters=self.n_gaussians[i],
                             seed=self.seed,
                             name=gmm_name )( stop_grad_layer )
                self.gmm_dict.update( {layer_name: gmm_name} )
                self.gmm_layers.append( gmm_name )

            else:
                raise ValueError( f'the modeled layer tensor shape must be either 2 or 4, but got: {len(x.shape)}' )

            if self.set_gmm_layer_as_output or self.training_method == 'generative':
                outputs_array.append( layer )

            if self.set_gmm_activation_layer_as_output or self.training_method == 'discriminative':
                get_custom_objects().update( {'gmm_bayes_activation': Activation( gmm_bayes_activation )} )
                x = Activation( gmm_bayes_activation, name='activation_' + gmm_name )( layer )
                self.gmm_activations_dict.update( {gmm_name: 'activation_' + gmm_name} )
                layers_array.append( x )

            if self.set_gmm_activation_layer_as_output:
                outputs_array.append( x )

        return outputs_array, layers_array

    def _build_layers(self):
        """Builds all layers
        """
        z, y, x = self.input_shape
        self.input_layer = Input( shape=(z, y, x) )

        # Preparing CNN
        encoded = self._build_encoder_layers( self.input_layer, name=self.network_name, gmm_layers=self.modeled_layers )

        # Preparing the combined model output Output layer parametrizes a Gaussian Mixture Density.
        if self.add_top:
            output1 = self._build_classifier_layers( encoded[0] )
        else:
            output1 = encoded[0]

        self.network_output_layer_name = output1.name.rsplit( '/', 1 )[0]

        outputs, encoded2 = self._build_gmm_layers( encoded[1:] )
        if self.training_method == 'discriminative':
            output2 = self._build_gmm_classifier_layers( encoded2 )
            outputs = outputs + output2

        if self.set_classification_layer_as_output:
            outputs.insert( 0, output1 )

        self.output_layers = outputs

    def compile_model(self, options=None):
        makedir( self.model_path )
        self.keras_model.compile( optimizer=self.optimizer, loss=self.losses, metrics=self.metrics_dict,
                                  options=options )

    def build_model(self):
        """Builds and compiles the model.
        """
        self._build_layers()
        self.keras_model = Model( self.input_layer, self.output_layers )

        variable_losses = {}
        metrics_dict = {}

        # Building losses
        for output_layer in self.keras_model._output_layers:
            layer_name = output_layer.name
            layer_type = type( output_layer ).__name__
            if layer_type == 'Dense':
                key = layer_name
                variable_losses[key] = categorical_crossentropy
                metrics_dict[key] = 'accuracy'
            elif layer_type == 'GMM' and self.training_method != 'discriminative':
                key = layer_name
                variable_losses[key] = build_gmm_loss( self.hyper_lambda )

        self.losses = variable_losses
        self.metrics_dict = metrics_dict

    def summary(self):
        """Print summary of model to stdout.
        """
        self.keras_model.summary()

    def save_weights_to_file(self, path):
        self.keras_model.save_weights( path )
        print( 'Saved weights to file' )

    def load_weights_from_file(self, path):
        """Load the weights of the model from a file.
        Args:
            path (str): Path to the weights file.
        """
        self.keras_model.load_weights( path, by_name=True )
        print( 'Loaded weights from file.' )

    def load_model(self, config, weights_dir=None):

        # with CustomObjectScope( {'GMM': GMM, 'gmm_bayes_activation': gmm_bayes_activation} ):
        #     self.keras_model = load_model( keras_model_path )

        self.__init__( n_gaussians=config['n_gaussians'],
                       input_shape=config['input_shape'],
                       n_classes=config['n_classes'],
                       seed=config['seed'],
                       hyper_lambda=config['hyper_lambda'],
                       optimizer=config['optimizer'],
                       saving_dir=config['model_path'],
                       weights_name_format=config['weights_name_format'],
                       training_method=config['training_method'],
                       network_name=config['network_name'],
                       set_gmm_activation_layer_as_output=config['set_gmm_activation_layer_as_output'],
                       set_gmm_layer_as_output=config['set_gmm_layer_as_output'],
                       set_classification_layer_as_output=config['set_classification_layer_as_output'],
                       freeze=config['freeze'],
                       layers_to_model=config['modeled_layers'],
                       weights_dir=config['weights_dir'],
                       add_top=config['add_top']
                       )
        self.build_model()
        if weights_dir is not None:
            self.load_weights_from_file(weights_dir)
        self.compile_model()

        # self.metrics_dict = config['metrics_dict']
        # self.gmm_dict = config['gmm_dict']
        # self.gmm_activations_dict = config['gmm_activations_dict']
        # self.classifiers_dict = config['classifiers_dict']
        # self.gmm_layers = config['gmm_layers']

    def fit(self, x=None, y=None, batch_size=None, epochs=1, validation_split=0.2, validation_data=None):

        callbacks = self._build_callbacks()

        if validation_data is not None:
            history = self.keras_model.fit( x=np.array( x ), y=y, batch_size=batch_size,
                                            epochs=epochs, callbacks=callbacks,
                                            validation_data=validation_data,
                                            shuffle=True )
        else:
            history = self.keras_model.fit( x=np.array( x ), y=y, batch_size=batch_size,
                                            epochs=epochs, callbacks=callbacks,
                                            validation_split=validation_split,
                                            shuffle=True )
        return history

    def fit_generator(self, datagen=None, x=None, y=None, batch_size=None, epochs=1, validation_data=None):

        callbacks = self._build_callbacks()

        if x is not None:
            history = self.keras_model.fit_generator( datagen.flow( x, y, batch_size=batch_size ),
                                                      epochs=epochs, callbacks=callbacks,
                                                      validation_data=validation_data )
        else:
            history = self.keras_model.fit_generator( generator=datagen,
                                                      steps_per_epoch=int(
                                                          datagen.image_filenames.shape[0] // batch_size ),
                                                      epochs=epochs,
                                                      callbacks=callbacks,
                                                      verbose=1,
                                                      validation_data=validation_data,
                                                      validation_steps=int(
                                                          validation_data.image_filenames.shape[0] // batch_size ) )
        return history

    def get_layer_predict(self, layer_name, x):
        get_output = K.function( [self.keras_model.layers[0].input, K.learning_phase()],
                                 [self.keras_model.get_layer( name=layer_name ).output] )
        return get_output( [x, 0] )[0]

    @staticmethod
    def _create_preds_dict(preds, output_layers):
        print( 'finish predict' )
        preds_dict = {'GMM': {}, 'classification': {}}

        for i, output_layer in enumerate( output_layers ):
            layer_type = type( output_layer ).__name__
            if layer_type == 'GMM':
                gmm_pred = preds[i]
                field_name = 'llr_' + output_layer.name
                preds_dict['GMM'][field_name] = gmm_pred

            elif layer_type == 'Activation':
                layer_name = output_layer.name.split( '_' )
                gmm_layer_name = ''
                for s in layer_name[1:]:
                    gmm_layer_name = gmm_layer_name + '_' + s
                gmm_layer_name = gmm_layer_name[1:]

                preds_dict['GMM'][gmm_layer_name] = preds[i]

            elif layer_type == 'Dense':
                preds_dict['classification'][output_layer.name] = preds[i]
            else:
                raise ValueError( 'predicted layer type has to be Dense or GMM' )
        return preds_dict

    def predict(self, x, batch_size=32):
        """Predict from model.
        Args:
            array (numpy.ndarray): Input array of shape (samples, z, y, x) and type float32.

        Returns:
            dictionary containing 2 dictionaries:
            1. GMM (dictionary) - containing the gmm preds where the key to each prediction is the layer name
            2. classification (dictionary) - containing classification preds of each layer (the key is the layer name)

            for classification: array size N x n_classes with its score.
            for gmm: array size N * Height * Width * num_gaussians with log_likelihood (log p(h|x)) for each layer.
        """
        output_layers = self.keras_model._output_layers.copy()
        preds = self.keras_model.predict( x, batch_size=batch_size )

        return self._create_preds_dict( preds, output_layers )

    def predict_generator(self, generator, steps=None):
        preds = self.keras_model.predict_generator( generator, steps )
        output_layers = self.keras_model._output_layers

        return self._create_preds_dict( preds, output_layers )

    @staticmethod
    def evaluate(preds, labels):
        """Computes the summary/ statistic results of the experiment.
        Input:
            preds(ndarray) - a score matrix of each example
            Labels - a (N*n_classes) vector of the N samples labels.

        Output:
            Summary - a dict contains the statistics of the experiment in the following fields:
                Error - a float indicates the error rate of the classification algorithm.
                Confusion_mat - the confusion matrix -  a matrix of size M*M (M– the number of classes)
                                where M(i,j) is the probability that an example with true label i is predicted
                                as label j
        """
        error = []
        for key in preds:
            N_Data = preds[key].shape[0]
            preds_ind = np.argmax( preds[key], axis=1 )
            labels_ind = np.argmax( labels, axis=1 )

            error_inds = np.where( preds_ind != labels_ind )
            N_error = np.asarray( error_inds ).shape[1]
            Error = round( (N_error / N_Data), 3 )

            print( f'The {key} test error: ', Error )
            error.append( Error )
        return error

    @staticmethod
    def get_confusion_matrix(preds, labels):
        if len( labels.shape ) != 1:
            labels = np.argmax( labels, axis=1 )
        preds_ind = np.argmax( preds, axis=1 )
        CM = confusion_matrix( labels, preds_ind )

        return CM

    def get_correspond_gmm_layer(self, layer_name):
        if layer_name not in self.gmm_dict:
            print( 'There is no gmm layer that correspond to ', layer_name )
            return None
        else:
            return self.gmm_dict[layer_name]

    def get_correspond_conv_layer(self, gmm_layer_name):
        corresponed_conv = None
        for conv_name, value_name in self.gmm_dict.items():
            if value_name == gmm_layer_name:
                corresponed_conv = conv_name
                break

        if corresponed_conv is None:
            ValueError( 'There is no conv layer that correspond to ', gmm_layer_name )
        else:
            return corresponed_conv

    def set_weights(self, gmms_params_dict):
        """for each gmm layer set the mean and std weights to be the mean and std of conv layer it modeled.
        Args:
            keras_model - keras model
            layer_gmm_params_dict - a dict containing the gmm count_layers names as keys. each key is a dict containing the keys
            'mu' and 'std' as it returns from 'calc_layer_mean_and_std()' func below. the values must be as the size of the
             channels of the conv layer (1, channels).
        Returns:
            keras_model - the same model after the weights has been initiated.
            """

        for gmm_layer_name in gmms_params_dict:
            layer = self.keras_model.get_layer( gmm_layer_name )
            weights = layer.get_weights()
            layer_type = type( layer ).__name__

            if layer_type == 'GMM':
                if 'mu' in gmms_params_dict[layer.name]:
                    weights[0] = np.ones_like( weights[0] ) * gmms_params_dict[layer.name]['mu']
                if 'std' in gmms_params_dict[layer.name]:
                    weights[1] = np.ones_like( weights[1] ) * gmms_params_dict[layer.name]['std']
                if 'prior' in gmms_params_dict[layer.name]:
                    weights[2] = np.ones_like( weights[2] ) * gmms_params_dict[layer.name]['prior']

                if self.network_output_layer_name in layer.name and self.training_method == 'discriminative':
                    classifier_layer_name = self.classifiers_dict[layer.name]
                    classifier_layer = self.keras_model.get_layer( classifier_layer_name )
                    classifier_weights = classifier_layer.get_weights()
                    classifier_weights[0] = np.ones_like( classifier_weights[0] ) * \
                                            gmms_params_dict[layer.name]['mu']
                    classifier_layer.set_weights( classifier_weights )

            elif layer_type == 'Dense' or layer_type == 'Conv2D':
                if 'w' in gmms_params_dict[layer.name]:
                    weights[0] = np.ones_like( weights[0] ) * gmms_params_dict[layer.name]['w']
                if 'b' in gmms_params_dict[layer.name]:
                    weights[1] = np.ones_like( weights[1] ) * gmms_params_dict[layer.name]['b']

            layer.set_weights( weights )

    def calc_modeled_layers_mean_and_std(self, data):
        """For each conv layer calc the mean and std of the output data

        mode - 1 - returns the mean of the data.
               2 - returns K examples randomly picked as representatives. """

        keras_model = self.keras_model

        layer_gmm_params = {}

        for layer_name, gmm_layer_name in self.gmm_dict.items():
            layer_gmm_params[gmm_layer_name] = {}
            K = keras_model.get_layer( gmm_layer_name ).n_clusters

            if layer_name == self.network_output_layer_name:
                x = np.zeros( (K, K) )
                np.fill_diagonal( x, 1 )
                layer_gmm_params[gmm_layer_name] = {}
                layer_gmm_params[gmm_layer_name].update( {'mu': x} )
                layer_gmm_params[gmm_layer_name].update( {'std': 0.1} )

            else:
                x = get_layer_output( keras_model, layer_name, data )
                # K = keras_model.get_layer( gmm_layer_name ).K

                if len( x.shape ) == 4:
                    B_dim = x.shape[0]
                    H_dim = x.shape[1]
                    W_dim = x.shape[2]
                    D_dim = x.shape[3]
                    n_samples = B_dim * H_dim * W_dim
                    x = np.reshape( x, (n_samples, D_dim) )
                elif len( x.shape ) == 2:
                    n_samples = x.shape[0]
                else:
                    raise TypeError( 'Layer type must be either Dense or Conv2D' )

                chosen_reps = np.random.choice( n_samples, K )
                mu = x[chosen_reps, :]
                std = np.std( x, axis=0 )
                layer_gmm_params[gmm_layer_name].update( {'mu': mu} )
                layer_gmm_params[gmm_layer_name].update( {'std': std} )

        return layer_gmm_params

    def set_gmm_classification_weights(self, mean=1, std=0.1):
        # out_layer = self.keras_model.get_layer(self.network_output_layer_name)
        gmm_layer = self.get_correspond_gmm_layer(self.network_output_layer_name)
        layer = self.keras_model.get_layer(gmm_layer)
        weights = layer.get_weights()

        # mean
        K = layer.n_clusters
        x = np.zeros((K, K))
        np.fill_diagonal(x, mean)
        # prior
        prior = 1/self.n_classes
        # Set gmm layer weights
        weights[0] = np.ones_like(weights[0]) * x  # mean
        weights[1] = np.ones_like(weights[1]) * std  # std
        weights[2] = np.ones_like(weights[2]) * prior  # alpha
        layer.set_weights(weights)

        gmm_classifier = self.classifiers_dict[gmm_layer]
        layer = self.keras_model.get_layer(gmm_classifier)
        weights = layer.get_weights()
        # set classifier weights
        weights[0] = np.ones_like(weights[0]) * x  # mean
        weights[1] = np.ones_like(weights[1]) * 0
        layer.set_weights(weights)

    def save_config(self):
        config = {
            'n_gaussians': self.n_gaussians,
            'model_path': self.model_path,
            'weights_name_format': self.weights_name_format,
            'optimizer': self.optimizer,
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'modeled_layers': self.modeled_layers,
            'training_method': self.training_method,
            'set_gmm_layer_as_output': self.set_gmm_layer_as_output,
            'set_gmm_activation_layer_as_output': self.set_gmm_activation_layer_as_output,
            'set_classification_layer_as_output': self.set_classification_layer_as_output,
            'network_name': self.network_name,
            'seed': self.seed,
            'hyper_lambda': self.hyper_lambda,
            'freeze': self.freeze,
            'metrics_dict': self.metrics_dict,
            'gmm_dict': self.gmm_dict,
            'gmm_activations_dict': self.gmm_activations_dict,
            'classifiers_dict': self.classifiers_dict,
            'gmm_layers': self.gmm_layers,
            'weights_dir': self.weights_dir,
            'add_top': self.add_top
        }

        save_to_file( file_dir=self.model_path,
                      objs_name=['config'],
                      objs=[config] )

