import numpy as np
import pandas as pd
from gensim.models import word2vec

def main():
    model = word2vec.Word2Vec.load("300features_40minwords_10context")

    # 整個Embedding Matrix的大小(16490, 300,) 16490個詞乘上300特徵
    model_shape = model.wv.syn0.shape

    # 單個詞的向量長度(300, )
    term_shape = model['good'].shape

    print('end')

if __name__ == '__main__':
    main()
