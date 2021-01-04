'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-12 18:02:09
@LastEditTime: 2019-08-18 18:20:27
@LastEditors: Please set LastEditors
'''
import pickle as pkl
import numpy as np
from keras.layers import Conv1D,Dense,Dropout,MaxPooling1D,Flatten,Activation,LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.wrappers import scikit_learn
from sklearn.model_selection import GridSearchCV,cross_val_predict,cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam
import tensorflow as tf
import os
from sklearn import metrics
from keras_cv_model import *
os.environ["CUDA_VISIBLE_DEVICES"]="2"
def preprocess_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)

    y = encoder.transform(labels).astype(np.int32)
    y = np_utils.to_categorical(y)

    return y

def create_model(optimizer='adam',n_filter=100,kernel_size=25,mp_size=20,dense_num=200,activation_f='relu'):
    model=Sequential()
    model.add(Conv1D(filters=n_filter,kernel_size=kernel_size,activation=activation_f,input_shape=(180,20)))
    model.add(MaxPooling1D(pool_size=mp_size))
    # model.add(Flatten())
    model.add(LSTM(512))
    model.add(Dense(dense_num,activation=activation_f))
    model.add(Dense(2,))
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
    model.summary()
    return model
# dataset=np.load('PDB14120CV.npy')
# n_sample=dataset.shape[0]
# dataset1=np.load('PDB2272test.npy')
# n_sampletest=dataset1.shape[0]
# # n_sample=500
# # y=np.array([1]*(n_sample//2)+[0]*(n_sample//2))
# y=np.array([0]*(n_sample//2)+[1]*(n_sample//2))
# ty=np.array([0]*1153+[1]*1119)
# indices=np.arange(n_sample)
# indicestest=np.arange(n_sampletest)
# # np.random.shuffle(indices)
# print(indices)
# print(indicestest)
# dataset=dataset[indices]
# dataset1=dataset1[indicestest]
# Y=y[indices]
# TY=ty[indicestest]
# model=create_model()
# # acc=keras_CV(dataset,Y,TY)
# acc=duliceshi01(dataset,dataset1,Y,TY)
# print(acc)
# np.savetxt('pdbpredcnntest.csv',np.array([test_labels,test_pred,test_pro01]).T,delimiter=',',fmt='%s')

# 以下交叉验证

dataset=np.load('PDB14120CV.npy')
n_sample=dataset.shape[0]
# dataset1=np.load('mytest02.npy')
# n_sampletest=dataset1.shape[0]
# n_sample=500
# y=np.array([1]*(n_sample//2)+[0]*(n_sample//2))
y=np.array([0]*(n_sample//2)+[1]*(n_sample//2))
# ty=np.array([0]*82+[1]*2628)
indices=np.arange(n_sample)
# indicestest=np.arange(n_sampletest)
np.random.shuffle(indices)
print(indices)
# print(indicestest)
dataset=dataset[indices]
# dataset1=dataset1[indicestest]
Y=y[indices]
# TY=ty[indicestest]
# model=create_model()
# acc=keras_CV(dataset,Y,TY)
acc,test_labels,test_pro01,test_pred=keras_CV(dataset,Y)
print(acc)


np.savetxt('pdbpredlstm512.csv',np.array([test_labels,test_pred,test_pro01]).T,delimiter=',',fmt='%s')



'''
model=scikit_learn.KerasClassifier(build_fn=create_model,epochs=15,batch_size=512)
pred = cross_val_predict(model,dataset,Y,cv=5,n_jobs=10)
print('/n')
print(pred)
print(metrics.accuracy_score(y,pred))
#kernel_size=[8,16,24,32,64]
#mp_size=[10,20,30,40]
#n_filter=[16,32,64,128,256,512]
#batch_size=[32,64,128,256,512,1024]
#dense_num=[20,60,80,100,200]
#activation_f=['relu']
#param_grid=dict(activation_f=activation_f)
#grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=4,cv=5)

#grid_result=grid.fit(dataset,Y)
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, std, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, std, param))
'''