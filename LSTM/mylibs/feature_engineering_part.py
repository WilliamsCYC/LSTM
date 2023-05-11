import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use('seaborn')


random_seed = 1


def heat_map(data, redundant_feature_list=None):
    """use redundant list to neglect some features you do not want"""
    if redundant_feature_list is not None:
        data = data.drop(redundent_feature_list, axis=1)
    plt.figure(figsize=(len(data.columns), len(data.columns)))
    corrMatrix = data.corr().round(2)
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


def plot_inertia(data):
    """plot k means elbow method"""
    df = data.copy()
    scaler = MinMaxScaler()
    for i in df.columns:
        df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
    '''first resample weekly average data, then transpose for k means algorithm'''
    df = df.resample('W-FRI').mean().T.dropna(axis=1)

    'calculate inertia for different number of clusters'
    n_clusters = range(2, len(df.index))
    inertia = []

    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=random_seed)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.plot(n_clusters, np.divide(inertia, inertia[0]))
    plt.hlines(0.1, n_clusters[0], n_clusters[-1], 'r', linestyles='dashed')
    plt.hlines(0.05, n_clusters[0], n_clusters[-1], 'r', linestyles='dashed')
    plt.xlabel('clusters')
    plt.ylabel('relative inertia')
    plt.legend(['inertia', '10% relative inertia', '5% relative inertia'])


def get_clusters(data, n_clusters):
    """use k means to cluster"""
    """split training and testing data set to prevent data leakage"""
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    train_ = train.copy()
    scaler = MinMaxScaler()
    for i in train.columns:
        train_[i] = scaler.fit_transform(train[i].values.reshape(-1, 1))
    train = train_.resample('W-FRI').mean().T.dropna(axis=1)

    train = pd.DataFrame(train.values, index=train.index)
    '''k means clustering'''
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_seed)
    kmeans.fit(train)
    labels = kmeans.predict(train)
    train = pd.DataFrame({'Cluster': labels,
                          'Feature Name': train.index}
                         ).sort_values(by=['Cluster'], axis=0)

    train = train.reset_index(drop=True)
    '''return the result of clustering'''
    return train


def random_choose_features(data, df_clusters):
    """use the clustering result of get_clusters, randomly choose one of the features in each cluster"""
    selected_feature_list = []
    for i in list(set(df_clusters['Cluster'])):
        random.seed(random_seed)
        selected_feature = random.choice(df_clusters[df_clusters['Cluster'] == i]['Feature Name'].values)
        selected_feature_list.append(selected_feature)
    '''return a dataframe that is ready to be fed into my predicting model'''
    return data[selected_feature_list].copy()


def pca_k(df, df_clusters):
    """use PCA method to decrease the dimension to one in each cluster"""
    data = df.copy()
    data.dropna(axis=0, inplace=True)
    data.drop('label', axis=1, inplace=True)
    X_train, X_test = train_test_split(data, test_size=0.2, shuffle=False)

    pca_k_df = pd.DataFrame(index=data.index)

    '''I want to keep the structure of dataframe'''
    for i in data.columns:
        scaler = StandardScaler()
        '''also transform X_train for PCA use'''
        X_train[i] = scaler.fit_transform(X_train[i].values.reshape(-1, 1))
        data[i] = scaler.transform(data[i].values.reshape(-1, 1))
    '''for each cluster'''
    for i in list(set(df_clusters['Cluster'])):
        '''get the feature names in each cluster'''
        cluster = df_clusters[df_clusters['Cluster'] == i]
        feature_name = cluster['Feature Name']

        '''do PCA only when there are more than one feature'''
        if len(feature_name) >= 1:
            pca = PCA(n_components=1)
            pca.fit(X_train[feature_name])
            aux = pca.transform(data[feature_name])
            pca_k_df = pd.concat([pca_k_df, pd.DataFrame(aux, index=pca_k_df.index)], axis=1)
    '''return a dataframe that is ready to be fed into my predicting model'''
    return pca_k_df
