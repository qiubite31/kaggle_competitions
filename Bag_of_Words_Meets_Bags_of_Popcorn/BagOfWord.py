import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def clean_review_content(review):
    # 使用BeautifulSoup移除HTML標籤
    review = BeautifulSoup(review, "lxml").get_text()

    # 使用re移除符號
    review = re.sub('[^a-zA-Z]', ' ', review)

    # 將內容轉成小寫
    review = review.lower()

    # 簡單的使用空格作斷詞
    words = review.split()
    # print(words)

    # 使用NLTK的停用字詞集過濾停用詞(Stop Word)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # print(words)

    # 用空格區格合併字詞
    review = ' '.join(words)
    return review


def main():

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)

    train['clean_review'] = train['review'].apply(clean_review_content)

    # 可比對資料處理完前後的差異
    # print(train['review'][9])
    # print(train['clean_review'][9])

    # 建立一個CountVectorizer來計算Bag of Word
    vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    # fit後轉成numpy array
    train_data_features = vectorizer.fit_transform(train['clean_review'].tolist())
    train_data_features = train_data_features.toarray()

    # 可看特徵向量的矩陣型狀 (25000, 5000,)
    feature_shape = train_data_features.shape

    # 可看前5000字是哪5000字
    vocab = vectorizer.get_feature_names()

    # 合併看這5000字的詞頻數量
    terms_tf = zip(vocab, np.sum(train_data_features, axis=0))

    # 使用隨機森林建立一個分類器並輸入特徵矩陣學習
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_data_features, train["sentiment"])

    # 載入測試資料集
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'),
                       header=0, delimiter="\t", quoting=3)

    # 開始review的清理
    test['clean_review'] = test['review'].apply(clean_review_content)

    # 轉換成矩陣
    test_data_features = vectorizer.fit_transform(test['clean_review'].tolist())
    test_data_features = test_data_features.toarray()

    # 使用分類器進行分類後，輸出判斷結果
    # 使用預設的處理方法，submit後分類為0.56908
    result = clf.predict(test_data_features)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

if __name__ == '__main__':
    main()


