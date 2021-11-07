#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
import pylab as pl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA as ICA
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split


# In[2]:


np.random.seed(42)


# In[3]:


hd=pd.read_csv("heart.csv")


# In[4]:


y=hd['target']
X=hd.drop(columns=['target'])


# In[5]:


mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)


# In[10]:


PCA_data = PCA(n_components = 6, whiten=False)
PCA_data.fit(X)
PCA_data_trans = PCA_data.transform(X)


# In[11]:


ICA_data = FastICA(n_components = 5)
ICA_data.fit(X)
ICA_data_trans = ICA_data.transform(X)


# In[12]:


RP_data = GaussianRandomProjection(n_components=2, eps=0.1)
RP_data_trans = RP_data.fit_transform(X)


# In[13]:


LDA_data = LinearDiscriminantAnalysis(n_components=1)
LDA_data_trans = LDA_data.fit_transform(X, y)


# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[15]:


clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1, momentum = 0.1, learning_rate_init = 0.1,max_iter=7000)


# In[16]:


plot_learning_curve(clf, "MLP using PCA learning curve", PCA_data_trans, y)


# In[17]:


plot_learning_curve(clf, "MLP using ICA learning curve", ICA_data_trans, y)


# In[18]:


plot_learning_curve(clf, "MLP using RP learning curve", RP_data_trans, y)


# In[19]:


plot_learning_curve(clf, "MLP using LDA learning curve", LDA_data_trans, y)


# In[20]:


clusterer = KMeans(n_clusters=10, random_state=10).fit(PCA_data_trans)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(PCA_data_trans)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using PCA transformed features(KMEANS)", X_df, y)


# In[21]:


clusterer = KMeans(n_clusters=16, random_state=10).fit(ICA_data_trans)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(ICA_data_trans)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using ICA transformed features(KMEANS)", X_df, y)


# In[22]:


clusterer = KMeans(n_clusters=5, random_state=10).fit(RP_data_trans)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(RP_data_trans)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using RP transformed features(KMEANS)", X_df, y)


# In[23]:


clusterer = KMeans(n_clusters=5, random_state=10).fit(LDA_data_trans)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(LDA_data_trans)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using RP transformed features(KMEANS)", X_df, y)


# In[24]:


EM_cluster = GaussianMixture(n_components=3).fit(PCA_data_trans)
EM = clusterer.labels_
X_df = pd.DataFrame(PCA_data_trans)
X_df[11] = EM
plot_learning_curve(clf, "MLP using PCA transformed features(EM)", X_df, y)


# In[25]:


EM_cluster = GaussianMixture(n_components=6).fit(ICA_data_trans)
EM = clusterer.labels_
X_df = pd.DataFrame(ICA_data_trans)
X_df[11] = EM
plot_learning_curve(clf, "MLP using ICA transformed features(EM)", X_df, y)


# In[26]:


EM_cluster = GaussianMixture(n_components=5).fit(RP_data_trans)
EM = clusterer.labels_
X_df = pd.DataFrame(RP_data_trans)
X_df[11] = EM
plot_learning_curve(clf, "MLP using RP transformed features(EM)", X_df, y)


# In[27]:


EM_cluster = GaussianMixture(n_components=8).fit(LDA_data_trans)
EM = clusterer.labels_
X_df = pd.DataFrame(LDA_data_trans)
X_df[11] = EM
plot_learning_curve(clf, "MLP using LDA transformed features(EM)", X_df, y)


# In[ ]:




