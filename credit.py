#!/usr/bin/env python
# coding: utf-8

# In[128]:


from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas 

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


# In[129]:


np.random.seed(42)


# In[130]:


ct=pd.read_csv("CT.txt")
y=ct['defaultPayment_label']
X=ct.drop(columns=['defaultPayment_label'])


# In[142]:


ct


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[21]:


train_data=x_train
train_data = train_data.values
mms = MinMaxScaler()
mms.fit(train_data)
data = mms.transform(train_data)
labels = y_train


# In[22]:


test_data  = x_test
test_data = test_data.values
mms2 = MinMaxScaler()
mms2.fit(test_data)
data_test = mms.transform(test_data)
labels_test = y_test
sample_size = 50


# In[145]:


datax=X.copy().values
datay=y.copy().values


# In[147]:


from sklearn.metrics import accuracy_score, homogeneity_score


# In[160]:


kmeans = KMeans(n_clusters=8).fit(datax)
accuracy_score((1-kmeans.labels_), datay)


# In[23]:


#Elbow for Kmeans
Sum_of_squared_distances = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('K_Means: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[24]:



#Elbow for Expectation Max
silhouette_score = []

K = range(2,20)
for k in K:
    gm = GaussianMixture(n_components= k, n_init=2, random_state=0).fit(data)
    gmPredicted = gm.predict(data)
    silhouette_score.append(metrics.silhouette_score(data, gmPredicted, metric='euclidean', sample_size=sample_size))
    
plt.plot(K, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Expectation Maximization: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[25]:


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s     %.2fs       %.3f        %.3f        %.3f        %.3f        %.3f         %.3f       %.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, estimator.predict(data)),
             metrics.completeness_score(labels, estimator.predict(data)),
             metrics.v_measure_score(labels, estimator.predict(data)),
             metrics.adjusted_rand_score(labels, estimator.predict(data)),
             metrics.adjusted_mutual_info_score(labels,  estimator.predict(data)),
             metrics.silhouette_score(data, estimator.predict(data),metric='euclidean',sample_size=sample_size),
             float(sum(estimator.predict(data) == labels))/float(len(labels))))


# In[26]:



n_digits_i = [10,20,30,40,50]
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=data)
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=data)


# In[27]:


PCA_data = PCA().fit(data)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(PCA_data.explained_variance_ratio_*100),'bx-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('PCA: Credit Card Dataset Explained Variance')
plt.show()


# In[28]:


PCA_data


# In[29]:



PCA_data = PCA(n_components = 5, whiten=False)
PCA_data.fit(data)
PCA_data_trans = PCA_data.transform(data)
PCA_data_trans_test = PCA_data.transform(data_test)

#Plots
per_var = np.round(PCA_data.explained_variance_ratio_* 100, decimals=1)
print("Original Credit Card Data Set Number of Rows and Columns:", data.shape)
print("Original Credit Card Data Set Number of Rows and Columns:", data_test.shape)
print("PCA Credit Card Train Data Set Number of Rows and Columns:", PCA_data_trans.shape)
print("PCA Credit Card Test Data Set Number of Rows and Columns:", PCA_data_trans_test.shape)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(8,6))
plt.scatter(PCA_data_trans[:,0],PCA_data_trans[:,1], c =y_train)
plt.title('PCA: Credit Card Dataset Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))


# In[39]:


def run_ICA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pandas.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: "+ title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'bx-')
    plt.grid(False)
    plt.show()

run_ICA(data,labels,"ICA: Credit Card Dataset Avergae Kurtosis")


# In[40]:


ICA_data = FastICA(n_components = 25)
ICA_data.fit(data)
ICA_data_trans = ICA_data.transform(data)
ICA_data_trans_test = ICA_data.transform(data_test)


# In[41]:


RP_data = GaussianRandomProjection(n_components=5, eps=0.1)
RP_data_trans = RP_data.fit_transform(data)
RP_data_trans_test = RP_data.fit_transform(data_test)

plt.figure(figsize=(8,6))
plt.scatter(RP_data_trans[:,0],RP_data_trans[:,1], c = y_train)
plt.title('RP: Credit Card Dataset Graph')
plt.xlabel('RP1')
plt.ylabel('RP2')


# In[42]:



# Create and run an LDA
LDA_data = LinearDiscriminantAnalysis(n_components=1)
LDA_data_trans = LDA_data.fit_transform(data, labels)
LDA_data_trans_test = LDA_data.fit_transform(data_test, labels_test)


plt.figure(figsize=(8,6))
plt.scatter(LDA_data_trans[:,0], labels, c = y_train)
plt.title('LDA: Credit Card Dataset Graph')
plt.xlabel('LDA1')


# In[43]:



#Elbow for Kmeans(LDA)
Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(LDA_data_trans)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('K_Means/LDA: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[44]:


#Elbow for Kmeans(PCA)
Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(PCA_data_trans)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('K_Means/PCA: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[45]:


#Elbow for Kmeans(ICA)
Sum_of_squared_distances = []
K = range(1,40)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(ICA_data_trans)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('K_Means/ICA: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[46]:


#Elbow for Kmeans(RP)
Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(RP_data_trans)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('K_Means/RP: Credit Card Data Set Elbow Method For Optimal k')
plt.show()


# In[47]:



#Elbow for Expectation Max(LDA)
silhouette_score = []

K = range(2,10)
for k in K:
    gm = GaussianMixture(n_components= k,random_state=0).fit(LDA_data_trans)
    gmPredicted = gm.predict(LDA_data_trans)
    silhouette_score.append(metrics.silhouette_score(LDA_data_trans, gmPredicted, metric='euclidean', sample_size=sample_size))
    
plt.plot(K, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Expectation Maximization/LDA: Adult Data Set Elbow Method For Optimal k')
plt.show()


# In[48]:



#Elbow for Expectation Max(PCA)
silhouette_score = []

K = range(2,15)
for k in K:
    gm = GaussianMixture(n_components= k, random_state=0).fit(PCA_data_trans)
    gmPredicted = gm.predict(PCA_data_trans)
    silhouette_score.append(metrics.silhouette_score(PCA_data_trans, gmPredicted, metric='euclidean', sample_size=sample_size))
    
plt.plot(K, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Expectation Maximization/PCA: Adult Data Set Elbow Method For Optimal k')
plt.show()


# In[49]:


silhouette_score = []

K = range(2,10)
for k in K:
    gm = GaussianMixture(n_components= k,random_state=0).fit(ICA_data_trans)
    gmPredicted = gm.predict(ICA_data_trans)
    silhouette_score.append(metrics.silhouette_score(ICA_data_trans, gmPredicted, metric='euclidean', sample_size=sample_size))
    
plt.plot(K, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Expectation Maximization/ICA: Adult Data Set Elbow Method For Optimal k')
plt.show()


# In[50]:


silhouette_score = []

K = range(2,10)
for k in K:
    gm = GaussianMixture(n_components= k, random_state=0).fit(RP_data_trans)
    gmPredicted = gm.predict(RP_data_trans)
    silhouette_score.append(metrics.silhouette_score(RP_data_trans, gmPredicted, metric='euclidean', sample_size=sample_size))
    
plt.plot(K, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Expectation Maximization/RP: Adult Data Set Elbow Method For Optimal k')
plt.show()


# In[51]:


n_digits_i = [10]
for i in range(1):
    print(bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++PCA", data=PCA_data_trans))
    
n_digits_i = [20]
for i in range(1):
    print(bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++ICA", data=ICA_data_trans))

n_digits_i = [5]
for i in range(1):
    print(bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++RP", data=RP_data_trans))
    
n_digits_i = [5]
for i in range(1):
    print(bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++lda", data=LDA_data_trans))


# In[52]:



n_digits_i = [10]
for i in range(1):
  print(bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixturePCA", data=PCA_data_trans))
n_digits_i = [2]
for i in range(1):
  bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixtureICA", data=ICA_data_trans)
n_digits_i = [4]
for i in range(1):
  print(bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixtureRP", data=RP_data_trans))
n_digits_i = [4]
for i in range(1):
  print(bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixtureLDA", data=LDA_data_trans))


# In[53]:


data_nn = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4,), random_state=1, momentum = 0.1, learning_rate_init = 0.1)


# In[54]:


import timeit


# In[55]:


for i in range(0,2):
    print(i)


# In[ ]:



# train_acc=[]
# test_acc=[]
# for n in range(1,30):
#     PCA_data = PCA(n_components = n, whiten=False)
#     PCA_data.fit(data)
#     PCA_data_trans = PCA_data.transform(data)
#     PCA_data_trans_test = PCA_data.transform(data_test)
#     data_nn.fit(PCA_data_trans, labels)
#     data_train_pred_PCA = data_nn.predict(PCA_data_trans)
#     data_test_pred_PCA = data_nn.predict(PCA_data_trans_test)
#     a=float(sum(data_train_pred_PCA == labels))/float(len(labels))
#     b=float(sum(data_test_pred_PCA == labels_test))/float(len(labels_test))
#     train_acc.append(a)
#     test_acc.append(b)
# print(train_acc)
# print(test_acc)


# In[56]:


plt.plot(test_acc,'blue')
plt.plot(train_acc,'black')
plt.xlabel('Number of Components')
plt.ylabel('Test accuracy')
plt.title('Accuracy vs Number of Components')
plt.show()


# ![image.png](attachment:image.png)

# In[57]:


# Origin train
start_time=timeit.default_timer()
data_nn.fit(data, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred = data_nn.predict(data)
print(float(sum(data_train_pred == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[58]:


# Origin test
data_test_pred = data_nn.predict(data_test)
float(sum(data_test_pred == labels_test))/float(len(labels_test))


# In[59]:


#PCA Train

start_time=timeit.default_timer()
data_nn.fit(PCA_data_trans, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_PCA = data_nn.predict(PCA_data_trans)
print(float(sum(data_train_pred_PCA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# pca 5

# In[60]:


#PCA Test
data_test_pred_PCA = data_nn.predict(PCA_data_trans_test)
float(sum(data_test_pred_PCA == labels_test))/float(len(labels_test))


# In[61]:



# train_acc1=[]
# test_acc1=[]
# for n in range(1,30):
#     ICA_data = FastICA(n_components = n)
#     ICA_data.fit(data)
#     ICA_data_trans = ICA_data.transform(data)
#     ICA_data_trans_test = ICA_data.transform(data_test)
   
#     data_nn.fit(ICA_data_trans, labels) 
#     data_train_pred_ICA = data_nn.predict(ICA_data_trans)
#     data_test_pred_ICA = data_nn.predict(ICA_data_trans_test)
#     a=float(sum(data_train_pred_ICA == labels))/float(len(labels))
#     b=float(sum(data_test_pred_ICA == labels_test))/float(len(labels_test))
#     train_acc1.append(a)
#     test_acc1.append(b)
# print(train_acc1)
# print(test_acc1)


# In[62]:


#ICA Train
start_time=timeit.default_timer()
data_nn.fit(ICA_data_trans, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_ICA = data_nn.predict(ICA_data_trans)
print(float(sum(data_train_pred_ICA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[63]:



#ICA Test
data_test_pred_ICA = data_nn.predict(ICA_data_trans_test)
float(sum(data_test_pred_ICA == labels_test))/float(len(labels_test))


# In[64]:


#RP Train
start_time=timeit.default_timer()
data_nn.fit(RP_data_trans, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_RP = data_nn.predict(RP_data_trans)
print(float(sum(data_train_pred_RP == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[65]:



#RP Test
data_test_pred_RP = data_nn.predict(RP_data_trans_test)
float(sum(data_test_pred_RP == labels_test))/float(len(labels_test))


# In[66]:


#LDA Train
start_time=timeit.default_timer()
data_nn.fit(LDA_data_trans, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_LDA = data_nn.predict(LDA_data_trans)
print(float(sum(data_train_pred_LDA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[67]:



#LDA Test
data_test_pred_LDA = data_nn.predict(LDA_data_trans_test)
float(sum(data_test_pred_LDA == labels_test))/float(len(labels_test))


# In[68]:


#New Data Frames
pca_train_dataframe = pandas.DataFrame(PCA_data_trans)
pca_test_dataframe = pandas.DataFrame(PCA_data_trans_test)

ica_train_dataframe = pandas.DataFrame(ICA_data_trans)
ica_test_dataframe = pandas.DataFrame(ICA_data_trans_test)

rp_train_dataframe = pandas.DataFrame(RP_data_trans)
rp_test_dataframe = pandas.DataFrame(RP_data_trans_test)

lda_train_dataframe = pandas.DataFrame(LDA_data_trans)
lda_test_dataframe = pandas.DataFrame(LDA_data_trans_test)


# In[69]:


kmeans_cluster = KMeans(n_clusters=10)
pca_train_dataframe["cluster"] = kmeans_cluster.fit_predict(pca_train_dataframe[pca_train_dataframe.columns[0:]])
pca_test_dataframe["cluster"] = kmeans_cluster.fit_predict(pca_test_dataframe[pca_test_dataframe.columns[0:]])


# In[70]:


kmeans_cluster = KMeans(n_clusters=20)
ica_train_dataframe["cluster"] = kmeans_cluster.fit_predict(ica_train_dataframe[ica_train_dataframe.columns[0:]])
ica_test_dataframe ["cluster"] = kmeans_cluster.fit_predict(ica_test_dataframe [ica_test_dataframe.columns[0:]])


# In[71]:


kmeans_cluster = KMeans(n_clusters=5)
rp_train_dataframe["cluster"] = kmeans_cluster.fit_predict(rp_train_dataframe[rp_train_dataframe.columns[0:]])
rp_test_dataframe ["cluster"] = kmeans_cluster.fit_predict(rp_test_dataframe [rp_test_dataframe.columns[0:]])


# In[72]:


kmeans_cluster = KMeans(n_clusters=5)
lda_train_dataframe["cluster"] = kmeans_cluster.fit_predict(lda_train_dataframe[lda_train_dataframe.columns[0:]])
lda_test_dataframe["cluster"] = kmeans_cluster.fit_predict(lda_test_dataframe [lda_test_dataframe.columns[0:]])


# In[73]:


data_nn = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4,), random_state=1, momentum = 0.1, learning_rate_init = 0.1)


# In[117]:


#PCA Train
start_time=timeit.default_timer()
data_nn.fit(pca_train_dataframe, labels)
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_PCA = data_nn.predict(pca_train_dataframe)
print(float(sum(data_train_pred_PCA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[75]:


#PCA Test
data_test_pred_PCA = data_nn.predict(pca_test_dataframe)
float(sum(data_test_pred_PCA == labels_test))/float(len(labels_test))


# In[118]:


#ICA Train
start_time=timeit.default_timer()
data_nn.fit(ica_train_dataframe, labels) 
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_ICA = data_nn.predict(ica_train_dataframe)
print(float(sum(data_train_pred_ICA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[77]:


#ICA Test
data_test_pred_ICA = data_nn.predict(ica_test_dataframe)
float(sum(data_test_pred_ICA == labels_test))/float(len(labels_test))


# In[120]:


#RP Train
start_time=timeit.default_timer()
data_nn.fit(rp_train_dataframe, labels) 
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_RP = data_nn.predict(rp_train_dataframe)
print(float(sum(data_train_pred_RP == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[79]:


#RP Test
data_test_pred_RP = data_nn.predict(rp_test_dataframe)
float(sum(data_test_pred_RP == labels_test))/float(len(labels_test))


# In[121]:


#LDA Train
start_time=timeit.default_timer()
data_nn.fit(lda_train_dataframe, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_LDA = data_nn.predict(lda_train_dataframe)
print(float(sum(data_train_pred_LDA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[81]:


#LDA Test
data_test_pred_LDA = data_nn.predict(lda_test_dataframe)
float(sum(data_test_pred_LDA == labels_test))/float(len(labels_test))


# In[82]:


pca_train_dataframe1 = pandas.DataFrame(PCA_data_trans)
pca_test_dataframe1 = pandas.DataFrame(PCA_data_trans_test)

ica_train_dataframe1 = pandas.DataFrame(ICA_data_trans)
ica_test_dataframe1 = pandas.DataFrame(ICA_data_trans_test)

rp_train_dataframe1 = pandas.DataFrame(RP_data_trans)
rp_test_dataframe1 = pandas.DataFrame(RP_data_trans_test)

lda_train_dataframe1 = pandas.DataFrame(LDA_data_trans)
lda_test_dataframe1 = pandas.DataFrame(LDA_data_trans_test)


# In[83]:


EM_cluster = GaussianMixture(n_components=10)
pca_train_dataframe1["cluster"] = EM_cluster.fit_predict(pca_train_dataframe1[pca_train_dataframe1.columns[0:]])
pca_test_dataframe1["cluster"] = EM_cluster.fit_predict(pca_test_dataframe1[pca_test_dataframe1.columns[0:]])
EM_cluster = GaussianMixture(n_components=2)
ica_train_dataframe1["cluster"] = EM_cluster.fit_predict(ica_train_dataframe1[ica_train_dataframe1.columns[0:]])
ica_test_dataframe1["cluster"] = EM_cluster.fit_predict(ica_test_dataframe1 [ica_test_dataframe1.columns[0:]])
EM_cluster = GaussianMixture(n_components=4)
rp_train_dataframe1["cluster"] = EM_cluster.fit_predict(rp_train_dataframe1[rp_train_dataframe1.columns[0:]])
rp_test_dataframe1 ["cluster"] = EM_cluster.fit_predict(rp_test_dataframe1 [rp_test_dataframe1.columns[0:]])
EM_cluster = GaussianMixture(n_components=4)
lda_train_dataframe1["cluster"] = EM_cluster.fit_predict(lda_train_dataframe1[lda_train_dataframe1.columns[0:]])
lda_test_dataframe1["cluster"] = EM_cluster.fit_predict(lda_test_dataframe1 [lda_test_dataframe1.columns[0:]])


# In[122]:


#PCA Train
start_time=timeit.default_timer()
data_nn.fit(pca_train_dataframe1, labels) 
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_PCA = data_nn.predict(pca_train_dataframe1)
print(float(sum(data_train_pred_PCA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[85]:


#PCA Test
data_test_pred_PCA = data_nn.predict(pca_test_dataframe1)
float(sum(data_test_pred_PCA == labels_test))/float(len(labels_test))


# In[123]:


#ICA Train
start_time=timeit.default_timer()
data_nn.fit(ica_train_dataframe1, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_ICA = data_nn.predict(ica_train_dataframe1)
print(float(sum(data_train_pred_ICA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[87]:


#ICA Test
data_test_pred_ICA = data_nn.predict(ica_test_dataframe1)
float(sum(data_test_pred_ICA == labels_test))/float(len(labels_test))


# In[124]:


#RP Train
start_time=timeit.default_timer()
data_nn.fit(rp_train_dataframe1, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_RP = data_nn.predict(rp_train_dataframe1)
print(float(sum(data_train_pred_RP == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[89]:


#RP Test
data_test_pred_RP = data_nn.predict(rp_test_dataframe1)
float(sum(data_test_pred_RP == labels_test))/float(len(labels_test))


# In[125]:


#LDA Train
start_time=timeit.default_timer()
data_nn.fit(lda_train_dataframe1, labels)  
end_time=timeit.default_timer()
training_time=end_time-start_time
data_train_pred_LDA = data_nn.predict(lda_train_dataframe1)
print(float(sum(data_train_pred_LDA == labels))/float(len(labels)))
print("training_time:"+"{:.2f}".format(training_time))


# In[91]:


#LDA Test
data_test_pred_LDA = data_nn.predict(lda_test_dataframe1)
float(sum(data_test_pred_LDA == labels_test))/float(len(labels_test))


# In[ ]:




