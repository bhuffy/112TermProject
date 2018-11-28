# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# import numpy as np
# 
# import scipy.cluster.hierarchy as sch
# from sklearn.cluster import AgglomerativeClustering
# 
# from sklearn.cluster import KMeans
# 
# 
# # population = [2,15,49,69,19,50,86,100,42,35,66,10,57,92,13,24,101,4,40,46,20,71,78]
# # bins = [10,20,30,40,50,60,70,80,90,100,110,120,130]
# # 
# # plt.hist(population, bins, histtype="bar", rwidth=0.9)
# # 
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.title('Interesting title')
# # plt.legend()
# # plt.show()
# 
# 
# # create blobs
# data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)
# # create np array for data points
# points = data[0]
# # create scatter plot
# plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
# plt.xlim(-15,15)
# plt.ylim(-15,15)
# 
# ## k means
# 
# # # create kmeans object
# # kmeans = KMeans(n_clusters=4)
# # # fit kmeans object to data
# # kmeans.fit(points)
# # # print location of clusters learned by kmeans object
# # print(kmeans.cluster_centers_)
# # # save new clusters for chart
# # y_km = kmeans.fit_predict(points)
# # 
# # plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
# # plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
# # plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
# # plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='cyan')
# 
# 
# 
# 
# 
# ## Alternative to k-means by Agglomerative Clustering
# 
# # How it works:
# '''
# Rather than choosing a number of clusters and starting out with random centroids, we instead begin with every point in our dataset as a “cluster.” Then we find the two closest points and combine them into a cluster. Then, we find the next closest points, and those become a cluster. We repeat the process until we only have one big giant cluster.'''
# # create dendrogram
# dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
# # create clusters
# hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# # save clusters for chart
# y_hc = hc.fit_predict(points)
# 
# plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
# plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
# plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
# plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')
# 
# plt.show()

## Clustering

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)