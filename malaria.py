
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
import detectobjects as det
import os.path
from scipy import misc
import cv2
import readdata
import tensorflow as tf
from progress_bar import ProgressBar
import shapefeatures
from sklearn import ensemble


# In[2]:


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
opts['patch_stride_training'] = int(opts['patch_size'][0]*.25)


# In[3]:


trainfiles, valfiles, testfiles = readdata.create_sets(opts['img_dir'], train_set_proportion=.5, 
                                                  test_set_proportion=.5,
                                                  val_set_proportion=0)

train_y, train_X = readdata.create_patches(trainfiles, opts['annotation_dir'], opts['img_dir'], opts['patch_size'][0], opts['patch_stride_training'], grayscale=False, progressbar=True, downsample=opts['image_downsample'], objectclass=opts['object_class'], negative_discard_rate=opts['negative_training_discard_rate'])
test_y, test_X = readdata.create_patches(testfiles,  opts['annotation_dir'], opts['img_dir'], opts['patch_size'][0], opts['patch_stride_training'], grayscale=False, progressbar=True, downsample=opts['image_downsample'], objectclass=opts['object_class'], negative_discard_rate=opts['negative_training_discard_rate'])

# Cut down on disproportionately large numbers of negative patches
train_X, train_y = readdata.balance(train_X, train_y, mult_neg=200)


# In[4]:


# Create rotated and flipped versions of the positive patches
train_X, train_y = readdata.augment_positives(train_X, train_y)
test_X, test_y = readdata.augment_positives(test_X, test_y)


# In[5]:


print ('\n')
print ('%d positive training examples, %d negative training examples' % (sum(train_y), len(train_y)-sum(train_y)))
print ('%d positive testing examples, %d negative testing examples' % (sum(test_y), len(test_y)-sum(test_y)))
print ('%d patches (%.1f%% positive)' % (len(train_y)+len(test_y), 100.*((sum(train_y)+sum(test_y))/(len(train_y)+len(test_y)))))


# In[6]:


print(train_y.shape,train_X.shape)


# In[8]:


get_ipython().magic('pylab inline')


# In[9]:


N_samples_to_display = 10
pos_indices = np.where(train_y)[0]
pos_indices = pos_indices[np.random.permutation(len(pos_indices))]
for i in range(N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_pos = train_X[pos_indices[i],:,:,:]
    example_pos = np.swapaxes(example_pos,0,2)
    plt.imshow(example_pos)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    

neg_indices = np.where(train_y==0)[0]
neg_indices = neg_indices[np.random.permutation(len(neg_indices))]
for i in range(N_samples_to_display,2*N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_neg = train_X[neg_indices[i],:,:,:]
    example_neg = np.swapaxes(example_neg,0,2)
    plt.imshow(example_neg)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.gcf().set_size_inches(1.5*N_samples_to_display,3)
plt.savefig('figs/malaria_.png', bbox_inches='tight')


# In[10]:


train_X= np.rollaxis(train_X, 1, 4)  
print(train_X.shape)

test_X= np.rollaxis(test_X, 1, 4)  
test_X.shape

print(train_y)

train_y.shape=(train_X.shape[0],1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
train_y=enc.fit_transform(train_y).toarray()
print(train_y.shape)


test_y.shape=(test_X.shape[0],1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
test_y=enc.fit_transform(test_y).toarray()
print(test_y.shape)

print(test_y)


# In[4]:


import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Convolution2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,Concatenate
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.regularizers import l2


# In[5]:


WEIGHTS_PATH = 'squeezenet_weights.h5'

def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x


# In[6]:


def SqueezeNet(input_shape=None, input_tensor=None,pooling=None):
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
    


# In[24]:


import h5py
import numpy as np
from keras.preprocessing import image
import numpy as np
squeeze= SqueezeNet(input_shape=(20,20,3))
x=squeeze.output
#x = AveragePooling2D(pool_size=(2, 2), name='avgpool10')(x)
x= BatchNormalization(name="norm1")(x)
x = _fire(x, (32, 128, 128), name="fire4")
x = _fire(x, (32, 128, 128), name="fire5")
x = Flatten(name='flatten10')(x)
x= Dense(2)(x)
x = Activation("softmax", name='softmax')(x)
model=Model(input=squeeze.input,output=x)
#print(model.summary())
for layer in model.layers[:]:
    layer.trainable=False
print(model.summary())


# In[25]:


model_json = model.to_json()
with open("malaria.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


from keras import optimizers
optr=optimizers.Adam(lr=1e-4)


# In[ ]:


model.compile(optimizer=optr, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


#model.optimizer.lr=1e-4
model.fit(train_X, train_y, epochs = 4, batch_size = 256)


# In[ ]:


fileName="malaria.h5"
file = h5py.File(fileName,'w')
weight = model.get_weights()
for i in range(len(weight)):
    file.create_dataset('weight'+str(i),data=weight[i])
file.close()


# In[53]:


fileName="malaria.h5"
file=h5py.File(fileName,'r')
weight = []
for i in range(len(file.keys())):
    weight.append(file['weight'+str(i)][:])
model.set_weights(weight)


# In[ ]:


import time
start=time.time()
preds = model.evaluate(test_X, test_y)
end=time.time()
print((end-start)/test_y.shape[0])


# In[28]:


preds = model.evaluate(test_X, test_y)


# In[29]:


print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

print(len(preds))


# In[62]:


print(test_X.shape)


# In[63]:


np.save("dummy.npy",test_X[0,:,:,:])


# In[67]:


y_pred=model.predict(test_X)


# In[55]:


print(y_pred.shape)


# In[19]:


filename = 'model.h5'
f = h5py.File(filename, 'r+')
print(list(f))


# In[3]:


get_ipython().magic('pylab inline')
import pandas as pd
y_pred=np.load('prediction_malaria.npy')
print(y_pred.shape)
df=pd.read_csv('val.txt',sep=' ',header=None)
#name=df[0]
idx=df[1]
print(idx.shape)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(idx, y_pred[:,1])
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

precision, recall, thresholds = metrics.precision_recall_curve(idx, y_pred[:,1])
average_precision = metrics.average_precision_score(idx,y_pred[:, 1])

subplot(121)
plt.title('ROC: AUC = %0.2f'% roc_auc)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylim([-.05, 1.05])
plt.xlim([-.05, 1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

subplot(122)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall: AP={0:0.2f}'.format(average_precision))
plt.legend(loc="lower left")

plt.gcf().set_size_inches(10,4)
plt.savefig('figs/malaria_evaluation1.png', bbox_inches='tight')


# In[ ]:


false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test_y[:,1], y_pred[:,1])
true_positive_rate.shape, thresholds.shape
plt.plot(true_positive_rate, thresholds,label='True positive rate')
plt.plot(false_positive_rate, thresholds, label='False positive rate')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.savefig('figs/malaria_evaluation2.png', bbox_inches='tight')


# In[58]:


neg_indices = np.where(test_y[:,1]==0)[0]
neg_scores = y_pred[neg_indices,1]
neg_indices = neg_indices[neg_scores.argsort()]
neg_indices = neg_indices[::-1]

neg_scores = y_pred[neg_indices,1]

N_samples_to_display = 12
offset = 55
for i in range(N_samples_to_display,2*N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_neg = test_X[neg_indices[i+offset],:,:,:]
    #example_neg = np.swapaxes(example_neg,0,2)
    plt.imshow(example_neg)
    plt.title('%.3f' % neg_scores[i+offset])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    

plt.gcf().set_size_inches(1.5*N_samples_to_display,3) 

plt.savefig('figs/malaria-falsedetections.png', bbox_inches='tight')


# In[59]:


prob_range = [.95,1.]

tmp_scores = y_pred.copy()[:,1]
tmp_scores[tmp_scores<prob_range[0]] = -1
tmp_scores[tmp_scores>prob_range[1]] = -1

pos_indices = tmp_scores.argsort()
pos_indices = pos_indices[::-1]

N_samples_to_display = 12
offset = 0
for i in range(N_samples_to_display,2*N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_neg = test_X[pos_indices[i+offset],:,:,:]
    #example_neg = np.swapaxes(example_neg,0,2)
    plt.imshow(example_neg)
    plt.title('%.3f' % (tmp_scores[pos_indices[i+offset]]))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    

plt.gcf().set_size_inches(1.5*N_samples_to_display,3) 

plt.savefig('figs/malaria-detectedpatches.png', bbox_inches='tight')


# In[60]:


pos_indices = y_pred[:,1].argsort()

N_samples_to_display = 12

for i in range(N_samples_to_display,2*N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_neg = test_X[pos_indices[i],:,:,:]
    #example_neg = np.swapaxes(example_neg,0,2)
    plt.imshow(example_neg)
    plt.title('%.3f' % (y_pred[pos_indices[i],1]))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    

plt.gcf().set_size_inches(1.5*N_samples_to_display,3) 

plt.savefig('figs/malaria-lowprob.png', bbox_inches='tight')


# In[180]:


# serialize model to JSON
'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
fileName="model.h"
file = h5py.File(fileName,'w')
weight = model.get_weights()
for i in range(len(weight)):
    file.create_dataset('weight'+str(i),data=weight[i])
file.close()
'''


# In[40]:


# load json and create model
'''
from keras.models import model_from_json
import h5py
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()

#loaded_model = model_from_json(loaded_model_json, custom_objects={'x': x} )
'''
# load weights into new model
#loaded_model.load_weights("model.h5")
json_file.close()


# In[24]:


fileName="model.h5"
file=h5py.File(fileName,'r')
weight = []
for i in range(len(file.keys())):
    weight.append(file['weight'+str(i)][:])
model.set_weights(weight)


# In[ ]:


history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[21]:



#reload(det)
get_ipython().magic('pylab inline')
import detectobjects as det
fname = 'plasmodium-phone-0873.jpg'
imfile = opts['img_dir'] + fname
opts['detection_probability_threshold'] = 0.95

found = det.detect(imfile, model, opts)

im = misc.imread(imfile)

plt.box(False)
plt.xticks([])
plt.yticks([])

annofile = opts['annotation_dir'] + fname[:-3] + 'xml'
bboxes = readdata.get_bounding_boxes_for_single_image(annofile)
for bb in bboxes:
    bb = bb.astype(int)
    cv2.rectangle(im, (bb[0],bb[2]), (bb[1],bb[3]), (255,255,255), 6)  
for f in found:
     f = f.astype(int)
     cv2.rectangle(im, (f[0],f[1]), (f[2],f[3]), (255,0,0), 6)

plt.gcf().set_size_inches(10,10)
plt.title('Detected objects in %s' % (imfile))
plt.imshow(im)
plt.savefig('figs/malariadata.png', bbox_inches='tight')


# In[ ]:
# 
# total training time 43*5s
