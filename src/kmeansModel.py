# Steven 05/17/2020
# clustering model design
from time import time
import pandas as pd
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler  # StandardScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
# from sklearn.metrics import make_scorer
# from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors._nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt


def createKMeans(k=2):
    model = KMeans(n_clusters=k, random_state=0)
    # print(model,k)
    # print('k=',k)
    return model


def s_score_silhouette(estimator, X):
    labels_ = estimator.fit_predict(X)
    score = 0
    # print(X.shape)
    # print(X)
    actualK = len(list(set(labels_)))
    if actualK > 1:
        # print(labels_)
        score = silhouette_score(X, labels_, metric='euclidean')  # 'euclidean'
        # score = calinski_harabasz_score(X, labels_)
        # score = davies_bouldin_score(X, labels_)
    # print(score)
    return score


def squaredDistances(a, b):
    return np.sum((a - b)**2)


def calculateSSE2(data, labels, centroids):
    print(data.shape, type(data), centroids.shape)
    sse = 0
    for i, ct in enumerate(centroids):
        # print('i,ct=',i,ct)
        samples = []

        for k in range(data.shape[0]):
            label = labels[k]
            sample = data.iloc[k, :].values
            # print('sample,label=',i,sample,label)
            if label == i:
                # sse += squaredDistances(sample,ct)
                samples.append(sample)

        sse += squaredDistances(samples, ct)
    return sse


def calculateSSE(data, labels, centroids):
    # print(data.shape,type(data),centroids.shape,labels.shape)
    # print('labels=',labels)
    # data = data.to_numpy()#if dataframe
    sse = 0
    for i, ct in enumerate(centroids):
        # print('i,ct=',i,ct)
        # samples = data.iloc[np.where(labels == i)[0], :].values
        samples = data[np.where(labels == i)[0], :]
        sse += squaredDistances(samples, ct)
    return sse


def calculateDaviesBouldin(data, labels):
    return davies_bouldin_score(data, labels)


def getModelMeasure(data, labels):
    sse, dbValue, csm = 0, 0, 0
    # k = len(np.unique(labels))
    k = len(list(set(labels)))
    if k > 1:
        # print(data.shape,model.labels_)
        # csm = silhouette_score(data, labels, metric='euclidean')
        clf = NearestCentroid()
        clf.fit(data, labels)
        # print(clf.centroids_)
        sse = calculateSSE(data, labels, clf.centroids_)
        # dbValue = calculateDaviesBouldin(data,labels)

    sse = round(sse, 4)
    csm = round(csm, 4)
    dbValue = round(dbValue, 4)
    print('SSE=', sse, 'DB=', dbValue, 'CSM=', csm, 'clusters=', k)
    # print("Silhouette Coefficient: %0.3f" % csm)
    # print('clusters=',k)
    return sse, dbValue, csm, k


def preprocessingData(data, N=5):
    scaler = MinMaxScaler()  # StandardScaler() #
    # scaler.fit(data)
    # data = scaler.transform(data)
    data = scaler.fit_transform(data)
    return data


def KMeansModelTrain(dataName, data, N=10):
    data = preprocessingData(data)
    df = pd.DataFrame()
    print('datashape=', data.shape)
    columns = ['Dataset', 'Algorithm', 'K', 'tt(s)', 'SSE', 'DB', 'CSM']
    for i in range(2, N, 1):  # 2 N 1
        model = createKMeans(i)
        modelName = 'K-Means'

        t = time()
        model.fit(data)

        tt = round(time() - t, 4)
        print("\ndataSet:%s model:%s iter i=%d run in %.2fs" % (dataName, modelName, i, tt))
        sse, dbValue, csm, k = getModelMeasure(data, model.labels_)

        dbName = dataName + str(data.shape)
        line = pd.DataFrame([[dbName, modelName, k, tt, sse, dbValue, csm]], columns=columns)
        df = df.append(line, ignore_index=True)
        # print('cluster_labels=',np.unique(model.labels_))

    # df.to_csv(gSaveBase + dataName+'_' + modelName+'_result.csv',index=True)
    print('Train result:\n', df)
    plotModel(dataName, modelName, df)

    index, bestK = getBestkFromSse(dataName, modelName, df)
    bestLine = df.iloc[index, :]
    print('bestLine=', index, 'bestK=', bestK, 'df=\n', bestLine)
    return bestK


def plotModel(datasetName, modelName, df):  # sse sse gredient
    # df = df.loc[:,['K', 'tt(s)', 'SSE','DB','CSM']]
    # print(df.iloc[:,[0,1,2,4]]) #no db

    x = df.loc[:, ['K']].values  # K
    y = df.loc[:, ['SSE']].values  # SSE
    z = np.zeros((len(x)))
    for i in range(len(x) - 1):
        z[i + 1] = y[i] - y[i + 1]

    # plt.figure(figsize=(8,5))
    ax = plt.subplot(1, 1, 1)
    title = datasetName + '_' + modelName + '_SSE'
    plt.title(title)
    ax.plot(x, y, label='SSE', c='k', marker='o')
    ax.plot(x, z, label='sse decrease gradient', c='b', marker='.')
    ax.set_ylabel('SSE')
    ax.set_xlabel('K clusters')
    ax.legend()
    ax.grid()
    plt.xticks(np.arange(1, 12))
    # plt.savefig(gSaveBase+title+'.png')
    plt.show()


def getBestkFromSse(datasetName, modelName, df):  # sse gradient
    print('df=\n', df)
    x = df.loc[:, ['K']].values  # K
    y = df.loc[:, ['SSE']].values  # SSE
    z = np.zeros((len(x)))  # gradient

    for i in range(len(x) - 1):
        z[i + 1] = y[i] - y[i + 1]

    index = np.argmax(z)
    bestK = x[index][0]
    # print('z=',z,index,bestK)
    return index, bestK


def KMeansModel(k, data):
    data = preprocessingData(data)
    model = createKMeans(k)
    model.fit(data)

    clf = NearestCentroid()
    clf.fit(data, model.labels_)
    return k, clf.centroids_, model.labels_
