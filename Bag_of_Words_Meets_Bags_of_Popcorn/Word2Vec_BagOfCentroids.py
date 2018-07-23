# coding=UTF-8
import os
import re
import time
import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.externals import joblib


def train_cluster(model):
    start = time.time()

    # 整個Embedding Matrix的大小(16490, 300,) 16490個詞乘上300特徵
    model_shape = model.wv.syn0.shape

    # cluster設定每5個詞一個cluster
    num_clusters = model_shape[0] / 5

    # fit並predict訓練資料的群體
    kmeans_clustering = KMeans(n_clusters=int(num_clusters))
    idx = kmeans_clustering.fit_predict(model.wv.syn0)

    end = time.time()
    elapsed = end - start
    print("分群花費時間:", elapsed)

    # 輸出每個詞所屬的群體
    joblib.dump(idx, 'cluster_obj.pkl')

    return idx


def read_cluster_model():
    return joblib.load('cluster_obj.pkl')


def main():

    is_train = False
    model = word2vec.Word2Vec.load("300features_40minwords_10context")

    if is_train:
        # 訓練並輸出模型
        idx = train_cluster(model)
    else:
        # 讀取訓練完的模型
        idx = read_cluster_model()

    # 將每個詞與所屬群建dict
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # 將群轉為key，value合併成list
    centroid_word_map = dict()
    for word, cluster in word_centroid_map.items():
        centroid_word_map.setdefault(int(cluster), []).append(word)

    # 輸出前10群的有哪些詞
    for cluster in range(0, 10):
        print('第{}群'.format(cluster))
        print(centroid_word_map[cluster])

if __name__ == '__main__':
    main()
