import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score

TARGET = ['SalePrice']
ORDER_FEATURES = ['MoSold']


def clean_data(df, missing_val=None, select_col=None):
    numeric_features = df.describe().columns.tolist()
    numeric_features.remove('MoSold')
    numeric_features.remove('Id')
    numeric_features.remove('YrSold')

    df = df[numeric_features]
    if missing_val is not None:
        df = df[select_col]
        df = df.fillna(missing_val)

    df = df.dropna(axis='columns', how='any')
    return df


def main():
    raw_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'train.csv'))

    df = clean_data(raw_df)
    x_df = df.drop('SalePrice', axis=1)
    y_df = df['SalePrice']

    scaler = StandardScaler()
    scaler.fit(x_df)
    x = scaler.transform(x_df)
    y = y_df
    # y = y_df.as_matrix()

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True, stratify=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    rg = LinearRegression()
    rg.fit(X_train, y_train.as_matrix())

    train_score = r2_score(y_train, rg.predict(X_train))
    test_score = r2_score(y_test, rg.predict(X_test))

    rg.fit(x, y)
    train_mean_df = x_df.mean()
    test_raw_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))
    test_df = clean_data(test_raw_df, missing_val=train_mean_df, select_col=x_df.columns.tolist())
    # test_df = test_df.drop('SalePrice', axis=1)
    test_df_scaled = scaler.transform(test_df)

    result = rg.predict(test_df_scaled)

    submission = pd.DataFrame(data={"id": test_raw_df["Id"], "SalePrice": result})
    submission.to_csv("submission.csv", index=False)

    print('end')

if __name__ == '__main__':
    main()
