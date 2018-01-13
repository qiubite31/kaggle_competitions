import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # event_df = pd.read_csv(r'events_train.csv')
    # event_label = pd.read_csv(r'labels_train.csv')
    # event_df.to_hdf('store_event.h5', 'table', append=True)
    # event_label.to_hdf('store_label.h5', 'table', append=True)
    event_df = pd.read_hdf('store_event.h5', 'table')
    # #####event_label = pd.read_hdf('store_label.h5', 'table')

    # 看每一個user的資料筆數
    user_df = event_df.set_index(['user_id'])
    user_cnt_df = user_df.reset_index(drop=False)['user_id'].value_counts()
    user_cnt_df = user_cnt_df.reset_index(drop=False)
    user_cnt_df = user_cnt_df.rename(columns={'index': 'user', 'user_id': 'count'})
    # user_cnt_df.to_csv('user_cnt.csv')
    # print(event_df.loc[53621])

    # 看資料筆數分布
    data_cnt_df = user_cnt_df['count'].value_counts()
    data_cnt_df = data_cnt_df.reset_index(drop=False)
    data_cnt_df = data_cnt_df.rename(columns={'index': 'data_cnt'})
    print(data_cnt_df)
    data_cnt_df.hist(['count'])
    plt.show()

    # 濾除資料後的筆數分布
    filter_threshold = 1
    data_cnt_filter_df = data_cnt_df[data_cnt_df['count'] > 1]
    data_cnt_filter_df.hist(['count'])
    plt.show()

    '''
    # title_df = event_df[['title_id', 'watch_time']].set_index(['title_id'])
    title_df = event_df[['title_id', 'watch_time']]
    # print(title_df)
    titles = title_df['title_id'].unique()
    title_wt = dict()
    for title in titles:
        title_data = title_df[title_df['title_id'] == title]['watch_time']
        title_wt[title] = title_data
        title_data.plot(kind='bar')
    # print(event_df)
    # print(event_label)
    '''
    pass
if __name__ == '__main__':
    main()
