'''
@Description: In User Settings Edit
@Author: unicnn
@Date: 2019-08-18 15:41:41
@LastEditTime: 2019-08-18 18:19:28
@LastEditors: Please set LastEditors
'''
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model,Sequential,load_model
import numpy as np
from keras.layers import Conv1D,Dense,Dropout,MaxPooling1D,Flatten,Activation,LSTM
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
def keras_CV(dataset,label,cv=5):
    skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
    i=0
    result=[]

    test_labels=[]
    test_pro01=[]
    test_pred=[]
    for train,test in skf.split(dataset,label):
        # print(train)
        # print(type(train))
        train_x=dataset[train]
        train_y=label[train]
        print(train_y)
        # print(type(train_y))<class 'numpy.ndarray'>
        train_y=np_utils.to_categorical(train_y)
        print(train_y)
        print(type(train_y))
        test_x=dataset[test]
        test_y=label[test]
        print(test_y)
        test_labels.extend(test_y)
        test_y=np_utils.to_categorical(test_y)
        early_stop=EarlyStopping(patience=5,verbose=1,monitor='val_acc')
        checkpoint=ModelCheckpoint('cv_model_'+str(i),verbose=1,monitor='val_acc',save_best_only=True)
        model=create_model()
        # model.fit(train_x,train_y,validation_data=[test_x,test_y],epochs=30,batch_size=100,callbacks=[checkpoint,early_stop])
        model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=30, batch_size=500,
                  callbacks=[checkpoint, early_stop])
        model=load_model('cv_model_'+str(i))
        predict_y = model.predict(test_x)
        predict_y_new=[]
        for temp in predict_y:
            print(temp[1])
            predict_y_new.append(temp[1])
            print(type(temp[1]))

        test_pro01.extend(predict_y_new)
        list_num1 = []
        for j in predict_y_new:
            if j >= 0.5:
                # i = 0
                list_num1.append(1)
            else:
                list_num1.append(0)

        test_pred.extend(list_num1)
        loss,acc=model.evaluate(test_x,test_y)

        print(acc)
        result.append(acc)
        i+=1
    result=np.array(result)
    
    return result,test_labels,test_pro01,test_pred

#独立测试
def duliceshi(dataset,dataset1,label, labeltest):
    # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    # i = 0
    test_labels = []
    test_pro01 = []
    test_pred = []
    result = []
    train_x = dataset
    train_y = label
    train_y = np_utils.to_categorical(train_y)
    test_x = dataset1
    test_y = labeltest
    test_y = np_utils.to_categorical(test_y)

    early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_acc')
    checkpoint = ModelCheckpoint('cv_model_' + str(i), verbose=1, monitor='val_acc', save_best_only=True)
    model = create_model()
    model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=30, batch_size=500,
              callbacks=[checkpoint, early_stop])
    model = load_model('cv_model_' + str(i))
    predict_y = model.predict(test_x)
    predict_y_new = []
    for temp in predict_y:
        print(temp[1])
        predict_y_new.append(temp[1])
        print(type(temp[1]))

    test_pro01.extend(predict_y_new)
    list_num1 = []
    for i in predict_y_new:
        if i >= 0.5:
            i = 0
            list_num1.append(i)
        else:
            list_num1.append(1)

    test_pred.extend(list_num1)
    loss, acc = model.evaluate(test_x, test_y)

    print(acc)
    result.append(acc)
    result = np.array(result)

    return result, test_labels, test_pro01, test_pred
    # checkpoint = ModelCheckpoint('cv_model_' + 'mycnntest01', verbose=1, monitor='val_acc', save_best_only=True)
    # model = create_model()
    # # model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=30, batch_size=500,
    # #           callbacks=[checkpoint, early_stop])
    # model.fit(train_x, train_y, epochs=30, batch_size=200,
    #           callbacks=[checkpoint, early_stop])
    # model = load_model('cv_model_' + 'mycnn01')
    # predict_y = model.predict(test_x, batch_size=200)
    # loss, acc = model.evaluate(test_x, test_y,batch_size=200)
    # print(acc)
    # print(predict_y)
    # result.append(acc)
    # result = np.array(result)
    #
    # return result
def duliceshi01(dataset,dataset1,label, labeltest):
    # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    # i = 0
    result = []
    train_x = dataset
    train_y = label
    train_y = np_utils.to_categorical(train_y)
    test_x = dataset1
    test_y = labeltest
    test_y = np_utils.to_categorical(test_y)
    early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_acc')
    checkpoint = ModelCheckpoint('test_model_01', verbose=1, monitor='val_acc', save_best_only=True)
    model = create_model()
    # model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=30, batch_size=500,
    #           callbacks=[checkpoint, early_stop])
    model.fit(train_x, train_y, validation_data=[test_x, test_y],epochs=30, batch_size=200,
              callbacks=[checkpoint, early_stop])
    model = load_model('test_model_01')
    predict_y = model.predict(test_x)
    loss, acc = model.evaluate(test_x, test_y)
    print(acc)
    print(predict_y)
    result.append(acc)
    result = np.array(result)

    return result
    # for train, test in skf.split(dataset, label):
    #     train_x = dataset[train]
    #     train_y = label[train]
    #     train_y = np_utils.to_categorical(train_y)
    #     test_x = dataset[test]
    #     test_y = label[test]
    #     test_y = np_utils.to_categorical(test_y)
    #     early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_acc')
    #     checkpoint = ModelCheckpoint('cv_model_' + str(i), verbose=1, monitor='val_acc', save_best_only=True)
    #     model = create_model()
    #     model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=30, batch_size=100,
    #               callbacks=[checkpoint, early_stop])
    #     model = load_model('cv_model_' + str(i))
    #     loss, acc = model.evaluate(test_x, test_y)
    #     print(acc)
    #     result.append(acc)
    #     i += 1
    # result = np.array(result)
    #
    # return result