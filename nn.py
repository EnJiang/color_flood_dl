from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add, Permute, GlobalMaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import keras.backend as K


def resnet(input_shape, nb_actions):
    input_ = Input(shape=input_shape)

    x = Permute((2, 3, 1))(input_)

    # first two conv
    x = Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(
        filters=512, kernel_size=3, padding='same', activation='relu')(x)

    # residual conv
    residual_module_n = 30
    for _ in range(residual_module_n):
        shortcut = x
        x = Conv2D(
            filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(
            filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = add([x, shortcut])

    x = GlobalMaxPooling2D()(x)

    x = Dense(1024, activation='tanh')(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(nb_actions, activation='linear')(x)

    model = Model(input_, x, name="ResNet")
    return model