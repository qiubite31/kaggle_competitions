# coding=UTF-8
import os
import re
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def review_to_wordlist(review, remove_stopword=False):
    # 使用BeautifulSoup移除HTML標籤
    review = BeautifulSoup(review, "lxml").get_text()

    # 使用re移除符號
    review = re.sub('[^a-zA-Z]', ' ', review)

    # 將內容轉成小寫
    review = review.lower()

    # 簡單的使用空格作斷詞
    words = review.split()
    # print(words)

    if remove_stopword:
        # 使用NLTK的停用字詞集過濾停用詞(Stop Word)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # print(words)

    return words


def train_cluster(model, num_clusters):
    start = time.time()

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


def create_bag_of_centroids(wordlist, word_centroid_map):

    # 總共群數
    num_centroids = max(word_centroid_map.values()) + 1

    # 初始化向量 (3298, )
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # 用cluster數量當特徵數量，判斷每一個詞所屬群後，計算每群有多少字當作特徵
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


def main():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)

    is_train = False
    model = word2vec.Word2Vec.load("300features_40minwords_10context")

    # 整個Embedding Matrix的大小(16490, 300,) 16490個詞乘上300特徵
    model_shape = model.wv.syn0.shape

    # cluster設定每5個詞一個cluster
    num_clusters = model_shape[0] / 5

    if is_train:
        # 訓練並輸出模型
        idx = train_cluster(model, num_clusters)
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
    # for cluster in range(0, 10):
    #     print('{}'.format(cluster))
    #     print(centroid_word_map[cluster])

    # 該群與大自然戶外相關
    print(centroid_word_map[19])

    # 先初始化訓練資料向量矩陣
    train_centroids = np.zeros((train["review"].size, int(num_clusters)), dtype="float32")

    # 將訓練資料前處理後轉原向量矩陣
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopword=True))

    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # 先初始化測試資料向量矩陣
    test_centroids = np.zeros((test["review"].size, int(num_clusters)), dtype="float32")

    # 將測試資料前處理後轉原向量矩陣
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopword=True))

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # 使用隨機森林建立一個分類器並輸入特徵矩陣學習
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_centroids, train["sentiment"])

    # 使用分類器進行分類後，輸出判斷結果
    result = forest.predict(test_centroids)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)

if __name__ == '__main__':
    main()
