import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def clean_review_content(review):
    # 使用BeautifulSoup移除HTML標籤
    review = BeautifulSoup(review).get_text()

    # 使用re移除符號
    review = re.sub('[^a-zA-Z]', ' ', review)

    # 將內容轉成小寫
    review = review.lower()

    # 簡單的使用空格作斷詞
    words = review.split()
    # print(words)

    # 使用NLTK的字用詞集過濾停用詞(Stop Word)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # print(words)

    # print('here')
    review = ' '.join(words)
    return review


train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                    header=0, delimiter="\t", quoting=3)

train['review'] = train['review'].apply(clean_review_content)

print(train['review'][9])
print('end')

