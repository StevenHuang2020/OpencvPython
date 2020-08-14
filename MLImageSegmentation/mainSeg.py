#python3 Steven segmentation using ML
#part of pixels for training, another part for testing
import os,sys
sys.path.append('..')

import time
import cv2
import numpy as np
import argparse 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from ImageBase import *
from makeImageDataSet import loadDb,makeImageFeatures
from classiferModels import *
import pickle

def newModel():
    #--------------------clustering---------------#
    return createKMeans()
    #return createAgglomerate()
    #return createDBSCAN()
    #return createSpectralClustering()

    #--------------------classfier----------------#
    #return createRandomForestClf()
    #return createLogisticRegression()
    #return createRidgeClassifier()
    #return createSGDClassifier()
    #return createSVM_svc()
    #return createSVM_NuSVC()
    #return createSVM_LinearSVC()
    #return createKNeighborsClassifier()
    #return createRadiusNeighborsClassifier()
    #return createNearestCentroid()
    #return createGaussianProcessClassifier()
    #return createGaussianNB()
    #return createMultinomialNB()

    #return createComplementNB()
    #return createBernoulliNB()
    #return createCategoricalNB()
    #return createDecisionTreeClassifier()
    #return createMLPClassifier()
    #return createLabelPropagation()

def saveModel(model,fileName=r'.\models\randomForesrPed001'):
    pickle.dump(model, open(fileName,'wb'))
    
def loadModel(fileName=r'.\models\randomForesrPed001'):
    return pickle.load(open(fileName,'rb'))

def train(X,y,save=False):
    #print('X.shape=',X.shape)
    #print(X.head())
    #print('class of y=',np.unique(y))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # print('X_train.shape=',X_train.shape)
    # print('Y_train.shape=',Y_train.shape)
    # print('X_test.shape=',X_test.shape)
    # print('Y_test.shape=',Y_test.shape)
    
    #X_train = preprocessingData(X_train)
    
    t = time()
    model,modelName = newModel()
    model.fit(X_train,Y_train)
    tt = round(time()-t, 4)
    
    print('\nModel:%s  run in %.2fs'%(modelName,tt))
    if modelName == 'DBSCAN':
        prediction_test = model.fit_predict(X_test)
    else:
        prediction_test = model.predict(X_test)
    print('Accuracy=', metrics.accuracy_score(Y_test, prediction_test))
    
#------------------------------------------------#
#Model:K-Means  run in 2.35s
#Accuracy= 0.7303295786399666
#Model:RandomForestClassifier  run in 9.62s
#Accuracy= 0.9363704630788485
#Model:LogisticRegression  run in 5.73s
#Accuracy= 0.9020942845223195
#Model:RidgeClassifier  run in 0.42s
#Accuracy= 0.9045306633291614
#Model:SGDClassifier  run in 23.60s
#Accuracy= 0.9029787234042553
#Model:KNeighborsClassifier  run in 23.15s
#Accuracy= 0.9194659991656237
#Model:NearestCentroid  run in 0.22s
#Accuracy= 0.6792991239048811
#Model:GaussianNB  run in 0.30s
#Accuracy= 0.8026533166458073
#Model:BernoulliNB  run in 0.26s
#Accuracy= 0.9046975385899041
#Model:DecisionTreeClassifier  run in 8.85s
#Accuracy= 0.911322486441385
#Model:MLPClassifier  run in 41.60s
#Accuracy= 0.9140926157697121
#Model:DBSCAN  run in 77.26s
#Accuracy= 0.0
#------------------------------------------------#

    if 0: #only randomforest have
        importances = list(model.feature_importances_)
        features = list(X_train.columns)
        print('importances=',importances)
        print('features=',features)
        
        f_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(f_imp)
        
    if save:
        saveModel(model)
    
def testPredict(X,img):
    print('img.shape=',img.shape)
    print('X.shape=',X.shape)
    
    model = loadModel()
    seg = model.predict(X).reshape((img.shape))

    plt.imshow(seg,cmap='jet')
    plt.savefig(r'.\res\Predict.png')
    plt.show()
    
def testPredictNew(file):
    img = loadGrayImg(file)
    X = makeImageFeatures(img)
    print('img.shape=',img.shape)
    print('X.shape=',X.shape)
    
    model = loadModel()
    seg = model.predict(X).reshape((img.shape))

    plt.imshow(seg,cmap='jet')
    plt.savefig(r'.\res\Predict.png')
    plt.show()
    
def main():
    file = r'.\res\FudanPed00001.png'
    maskFile = r'.\res\FudanPed00001_mask.png'
    dbFile=r'.\res\FudanPed00001.csv'
    
    img = loadGrayImg(file)
    mask = loadGrayImg(maskFile)
    #makeDb(img,mask,dbFile)

    X,y = loadDb(dbFile)
    
    if 1:
        train(X,y)
    else:
        #testPredict(X,img)
        file = r'.\res\FudanPed00019.png'
        testPredictNew(file)
        
if __name__=='__main__':
    main()
    