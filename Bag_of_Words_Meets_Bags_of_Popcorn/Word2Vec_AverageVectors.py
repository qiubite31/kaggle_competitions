import os
import re
import logging
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec


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


def review_to_sentences(review, tokenizer, remove_stopword=False):

    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopword))

    return sentences


def review_preprocess():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'unlabeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)

    # 檢查不同資料集的資料數量
    # label_train_size = train['review'].size
    # test_size = test['review'].size
    # unlabel_train_size = unlabeled_train['review'].size

    # nltk.download()
    # 使用NLTK來斷句
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # 將每篇review斷句後作斷詞
    sentences = []

    for review in train['review']:
        sentences += review_to_sentences(review, tokenizer)

    for review in unlabeled_train['review']:
        sentences += review_to_sentences(review, tokenizer)


def train_wrod2vector():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 設定Word2Vector的參數
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # 開始使用word2vec訓練模型
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    model.init_sims(replace=True)

    # 輸出模型
    model_name = "300features_40minwords_10context"
    model.save(model_name)


def read_word2vector_model():
    model = word2vec.Word2Vec.load("300features_40minwords_10context")
    return model


def main():
    is_proprocess = False
    is_train = False

    if is_proprocess:
        review_preprocess()

    if is_train:
        # 訓練並輸出模型
        train_wrod2vector()
    else:
        # 讀取訓練完的模型
        model = read_word2vector_model()

    print(model.doesnt_match("man woman child kitchen".split()))
    print(model.doesnt_match("france england germany berlin".split()))
    print(model.doesnt_match("paris berlin london austria".split()))
    print(model.most_similar("man"))
    print(model.most_similar("queen"))
    print(model.most_similar("awful"))


if __name__ == '__main__':
    main()
