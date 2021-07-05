import keras 
from keras.models import Sequential, load_model, Model 
from keras.layers import Dense, Dropout, Flatten, Activation, concatenate, Input 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
#import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, recall_score, precision_score,f1_score,classification_report
from keras.metrics import categorical_accuracy
#from keras.layers.convolutional import Conv2D,MaxPooling2D
import livelossplot
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot
from sklearn import metrics
import random
#from ann_visualizer.visualize import ann_viz;
from collections import Counter
from imblearn.pipeline import make_pipeline
from keras.callbacks import Callback
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import preprocessing
from numpy import mean, std
import pydot
import xlrd
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from numpy import array, argmax
# fix random seed for reproducibility
#seed = 2019
#np.random.seed(seed)
import umap
import os

def train_model(Xtrain, ytrain):    
    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.10, shuffle = True, stratify = ytrain)
    
    print(y_train.shape,y_test.shape)
    print(X_train.shape,X_test.shape)
    model = Sequential()     #create seq model
    n_cols = X_train.shape[1]  #number of columns
    print(n_cols)
    
	#model.add(Dense(1,activation='sigmoid',input_shape=(n_cols,)))    #One input and output layer
    
    #add layers to model    
    model.add(Dense(n_cols, activation='relu',name ='layer-1',kernel_initializer='random_uniform', input_shape=(n_cols,)))
    model.add(Dropout(0.25))
    model.add(Dense(int(2/3*n_cols), activation='relu', name='layer-2',kernel_initializer='random_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(int((2/3*n_cols)/2), activation='relu',name='layer-3',kernel_initializer='random_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax',name ='output-layer'))
    
    opt = keras.optimizers.Adam(lr=0.004)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(X_train, y_train, batch_size = 32,epochs=100,validation_data = (X_test, y_test),
                        callbacks=[ModelCheckpoint('model_best.hdf5', save_best_only=True, monitor='val_acc', mode='max')],verbose=0)
    
    print("Avg. Training Accuracy:", np.mean(history.history["acc"]))
    print("Avg. Validation Accuracy:", np.mean(history.history["val_acc"]))
    print("Avg. Training loss:", np.mean(history.history["loss"]))
    print("Avg. Validation loss:", np.mean(history.history["val_loss"]))
    pyplot.plot(history.history['acc'], label='train')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy")
    pyplot.plot(history.history['val_acc'], label='validation')
    pyplot.show()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.show()
    pyplot.show()

    print("Finished training")
    model = load_model("model_best.hdf5")
    #filename = 'model' + '.hdf5'
    #model.save(filename)
    return model
#%%
if __name__ == "__main__":
    df = pd.read_csv('processed-data.csv')
    train_y = df[['RECIDIVE']].to_numpy()
    train_X = df.iloc[:,0:383].to_numpy() 

#%% #Exploration  
    labels=train_y.ravel()
    cdict={0:'green',1:'red'}
    labl={0:'No',1:'Yes'}
    marker={0:'*',1:'o'}
    alpha={0:.3, 1:.5}

#%% #t-SNE
    X_tsne = manifold.TSNE(n_components=2,n_iter=1000, perplexity=30,verbose=1).fit_transform(train_X)
    Xax=X_tsne[:,0]
    Yax=X_tsne[:,1]
    fig,ax=plt.subplots(figsize=(13,10))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],marker=marker[l],alpha=alpha[l])
    plt.xlabel("First Component",fontsize=14)
    plt.ylabel("Second Component",fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig("renn-tsne", format='eps', dpi=1000)
    
 #%% #PCA   
    pca = PCA(n_components = 100)
    pca.fit(train_X)
    X_pca = pca.transform(train_X)
    Xax=X_pca[:,0]
    Yax=X_pca[:,1]
    fig,ax=plt.subplots(figsize=(13,10))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
        label=labl[l],marker=marker[l],alpha=alpha[l])
    #ax.scatter(Xax[100],Yax[100],c='black',marker='o',s=100)
    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig("renn-pca.eps", format='eps', dpi=1000)
    print("Here we can see that we cannot apply PCA, correlations are non linear")

#%% #UMAP
    X_umap = umap.UMAP(n_components=2,n_neighbors=50).fit_transform(train_X)
    Xax=X_umap[:,0]
    Yax=X_umap[:,1]
    fig,ax=plt.subplots(figsize=(13,10))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],marker=marker[l],alpha=alpha[l])
    plt.xlabel("First Component",fontsize=14)
    plt.ylabel("Second Component",fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig("renn-umap", format='eps', dpi=1000)

#%% #Processing and building model
    yy = train_y.ravel()
    renn = RepeatedEditedNearestNeighbours(sampling_strategy='majority',kind_sel='all',return_indices=True)
    xxx,yy,ind_sam = renn.fit_resample(train_X, yy)
    yyy = yy.reshape(len(yy),1)
    print(sorted(Counter(yy).items()))
    #np.savetxt("ind_sam_enn.csv", ind_sam, delimiter=",", fmt='%d') 
    enn_ind = ind_sam
    #ENN_IDs = IDs.loc[ind_sam]
    #ENN_IDs.reset_index(drop=True,inplace=True)

    test_size = round((0.10 * xxx.shape[0])*0.5)
    ids1=np.where(yyy==0)
    ids2=np.where(yyy==1)
    ids1 = ids1[0]
    ids2 = ids2[0]

    #random.seed=2020
    test1=np.random.choice(ids1,test_size,replace=False)
    for temp in test1:
        ids1 = np.delete(ids1,np.where(ids1==temp))
    #random.seed=2020
    test2=np.random.choice(ids2,test_size,replace=False)
    for temp in test2:
        ids2 = np.delete(ids2,np.where(ids2==temp))
    test=np.concatenate([test1,test2])
    #test =np.unique(test)

    test_X = xxx[test]
    test_y = yyy[test]
    train_y = np.delete(yyy,test,axis=0)
    train_X = np.delete(xxx,test,axis=0)
    #test_IDs = ENN_IDs.loc[test]
    #train_IDs = ENN_IDs.drop(test)
    sk1_hot = OneHotEncoder(sparse=False,handle_unknown='ignore')
    train_yy = sk1_hot.fit_transform(train_y)
    
    print(np.unique(test_y, return_counts=True))
    print(np.unique(train_y, return_counts=True))
    
    #x_sample_ind = np.delete(ind_sam,test,axis=0)
    #np.savetxt('Cat-RENNtrain_X.csv', train_X, delimiter=",")
    #np.savetxt('Cat-RENNtest_X.csv', test_X, delimiter=",")
    #np.savetxt('Cat-RENNtrain_y.csv', train_y, delimiter=",")
    #np.savetxt('Cat-RENNtest_y.csv', test_y, delimiter=",")
    #np.save("Cat-cols_name",Cols)
    #train_IDs.to_csv('Cat-RENN' + 'train_IDs' + '.csv')    
    #test_IDs.to_csv('Cat-RENN' + 'test_IDs' + '.csv')    
    model = train_model(train_X,train_yy)
    print(model.predict(test_X)) 
