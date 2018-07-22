import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier


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


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    # Loop through the reviews
    for review in reviews:
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        counter = counter + 1
    return reviewFeatureVecs


def main():
    num_features = 300
    model = word2vec.Word2Vec.load("300features_40minwords_10context")

    # 整個Embedding Matrix的大小(16490, 300,) 16490個詞乘上300特徵
    model_shape = model.wv.syn0.shape

    # 單個詞的向量長度(300, )
    term_shape = model['good'].shape

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'),
                        header=0, delimiter="\t", quoting=3)

    # 前處理訓練資料集的原始資料
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopword=True))

    # 將訓練資料轉成特徵向量
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    # 前處理測試資料集的原始資料
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopword=True))

    # 將測試資料轉原特徵向量
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    # 使用隨機森林建立一個分類器並輸入特徵矩陣學習
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # 使用分類器進行分類後，輸出判斷結果
    # 使用預處理的方法，分數為0.83436
    result = forest.predict(testDataVecs)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)


if __name__ == '__main__':
    main()
