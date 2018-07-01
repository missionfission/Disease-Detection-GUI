

import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Convolution2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,Concatenate
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
import h5py
import numpy as np
from keras.preprocessing import image
import numpy as np
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.regularizers import l2



# In[11]:


WEIGHTS_PATH = 'squeezenet_weights.h5'

def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x


# In[12]:


def SqueezeNet(input_tensor=None, input_shape=None, pooling=None):
    #input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=20,data_format=K.image_data_format())
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="same")(x)

    x = _fire(x, (16, 64, 64), name="fire2")
    x = _fire(x, (16, 64, 64), name="fire3")

    model = Model(img_input, x, name="squeezenet")

    #model.load_weights('squeezenet_weights2.h5',by_name=True,skip_mismatch=True)
    return model
    


# In[13]:


def load_model(input):
    squeezenet_model = SqueezeNet(input_shape=input)
    x=squeezenet_model.output
    #x = AveragePooling2D(pool_size=(2, 2), name='avgpool10')(x)
    x= BatchNormalization(name="norm1")(x)
    x = _fire(x, (32, 128, 128), name="fire4")
    x = _fire(x, (32, 128, 128), name="fire5")

    x = Flatten(name='flatten10')(x)
    x= Dense(2)(x)
    x = Activation("softmax", name='softmax')(x)
    model=Model(input=squeezenet_model.input,output=x)
    return model
def load_model_weights(input,fileName):
    model=load_model(input)
    file=h5py.File(fileName,'r')
    weight = []
    found = det.detect(imfile, model, opts)
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    return model

def set_opts(name):
    if (name=="malaria"):
      opts = {'img_dir': 'data/plasmodium-phone-verified/',
        'models_dir': '../models/',
        'annotation_dir': 'data/plasmodium-phone-verified/',
        'detection_probability_threshold': 0.5,
        'detection_overlap_threshold': 0.3, 
        'gauss': 1,
        'patch_size': (40,40),
        'image_downsample' : 2,
        'detection_step': 5,
        'patch_creation_step': 40,
        'object_class': None,
        'negative_training_discard_rate': .9
       }
    if (name=="tuberculosis"):
            opts = {'img_dir': 'data/tuberculosis-subset2/',
                'annotation_dir': 'data/tuberculosis-subset2/',
                'detection_probability_threshold': 0.5,
                'detection_overlap_threshold': 0.3, 
                'gauss': 1,
                'patch_size': (160,160),
                'image_downsample' : 8,
                'detection_step': 5,
                'patch_creation_step': 40,
                'object_class': None,
                'negative_training_discard_rate': .9
               }
    if (name=="intestinal"):
        opts = {'img_dir': 'data/intestinalparasites/',
        'models_dir': '../models/',
        'annotation_dir': 'data/intestinalparasites/',
        'detection_probability_threshold': 0.9,
        'detection_overlap_threshold': 0.3, 
        'gauss': 1,
        'patch_size': (600,600),
        'image_downsample' : 20,
        'detection_step': 5,
        'patch_creation_step': 40,
        'object_class': None,
        'negative_training_discard_rate': .9
       }
    opts['patch_stride_training'] = int(opts['patch_size'][0]*.25)
    return opts


def set_params(name):
    if (name=="malaria"):
        fileName="malaria.h5"
        input_shape=(20,20,3)
    if (name=="tuberculosis"):
        fileName="tuberculosis.h5"
        input_shape=(20,20,3)
    if (name=="intestinal"):
        fileName="intestinal"
        input_shape=(30,30,3)
    return input_shape,fileName