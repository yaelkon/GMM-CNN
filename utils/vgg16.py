from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers


def build_vgg16(input_shape, batch_size, weight_decay=0.0005, n_classes=10):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    weight_decay = weight_decay

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape ,kernel_regularizer=regularizers.l2(weight_decay), batch_size=batch_size,name='conv2d_1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_3'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_4'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_5'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_6'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_7'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_8'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_9'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay), batch_size=batch_size,name='conv2d_10'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), batch_size=batch_size,name='conv2d_11'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_12'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same' ,kernel_regularizer=regularizers.l2(weight_decay),name='conv2d_13'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512 ,kernel_regularizer=regularizers.l2(weight_decay),name='dense'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', name='classification'))

    return model