import numpy as np
import pandas as pd
from sklearn import preprocessing


def drop_columns(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # 不要な列の削除
    drop_columns = [
        "id",  # 普通に不要 (dfならindexで同じ値が取れる)
        "thumbnail_url",  # もしかしたらnanかどうかが利く？
        "amenities",  # 辞書型なのかリスト型なのか…… 扱いづらいため削除
        "description",  # 英文の説明 解析できたら強そうだけど諦め
        "first_review",  # レビューの日付 使いどころ不明
        "host_since",  # ホスト登録日 使いどころ不明
        "last_review",  # レビューの日付 使いどころ不明
        "name",  # 宿名 使いどころ不明
        "neighbourhood",  # 近隣情報 価値はありそうだけど面倒なので削除
        # "property_type",  # 物件の種類 整理したら使えそう
        "zipcode",  # 郵便番号 必要か？
    ]
    df_train.drop(drop_columns, axis=1, inplace=True)
    df_test.drop(drop_columns, axis=1, inplace=True)
    return df_train, df_test


# テキスト変換
def text_to_num(df: pd.DataFrame):
    # 数値情報テキストを数値に変換
    df["host_response_rate"] = df["host_response_rate"].str.replace("%", "").astype("float")

    # t-f表記を0-1に変換
    target = [
        "cleaning_fee",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]
    df[target] = df[target].replace({"f": False, "t": True})
    return df


# 欠損値処理
def fillna(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    null_enbedd_dict = {
        "bathrooms": 0,
        "bedrooms": 0,
        "beds": 1,
        "host_has_profile_pic": False,
        "host_identity_verified": False,
        "host_response_rate": df["host_response_rate"].median(),  # 中央値
        "review_scores_rating": df["review_scores_rating"].mean(),  # 平均値
    }
    df_train.fillna(null_enbedd_dict, inplace=True)
    df_test.fillna(null_enbedd_dict, inplace=True)
    return df_train, df_test


# 正規化・正則化処理
def normalize(df: pd.DataFrame):
    # 正規化処理
    df["host_response_rate"] = df["host_response_rate"] / 100  # 0-100%の範囲なので100で割る
    df["review_scoresrating"] = df["review_scores_rating"] / 100  # 0-100点の範囲なので100で割る

    df_train = df[df['y'].notna()]
    df_test = df[df['y'].isna()]

    # 正則化処理
    # 下記の項目を正則化する

    # 正則化処理
    # 下記の項目を正則化する
    target = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "latitude",
        "longitude",
        "number_of_reviews",
    ]
    for t in target:
        scaler = preprocessing.StandardScaler()
        df = df_train[[t]].copy()
        df_train[t] = scaler.fit_transform(df)
        df = df_test[[t]].copy()
        df_test[t] = scaler.transform(df)

    # y列も正則化し、スケーラを保存しておく
    y_scaler = preprocessing.StandardScaler()

    df = df_train[['y']].copy()
    df_train['y'] = y_scaler.fit_transform(df)

    return df_train, df_test, y_scaler


def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # , 'accommodates', 'amenities', 'bathrooms', 'bed_type', 'bedrooms',
    #   'beds', 'cancellation_policy', 'city', 'cleaning_fee', 'description',
    #   'first_review', 'host_has_profile_pic', 'host_identity_verified',
    #   'host_response_rate', 'host_since', 'instant_bookable', 'last_review',
    #   'latitude', 'longitude', 'name', 'neighbourhood', 'number_of_reviews',
    #   'property_type', 'review_scores_rating', 'room_type', 'thumbnail_url',
    #   'zipcode', 'y'

    # 不要な列の削除
    df_train, df_test = drop_columns(df_train, df_test)

    # テキスト変換
    df_train = text_to_num(df_train)
    df_test = text_to_num(df_test)

    # 欠損値処理 (基本的に妥当と思われる定数埋め)
    df_train, df_test = fillna(df_train, df_test)

    # カテゴリ変数はone-hotエンコーディングを行う
    # get_dummiesはカテゴリ変数をbool値(=0 or 1の数値)に変換するが、値が2種類しかないものでも2列に分割されてしまうので注意 (この例では対策をしていない)
    # df = pd.get_dummies(df)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    # 次の項目をone-hotエンコーディングする
    #   bed_type
    #   city
    #   cancellation_policy
    #   property_type
    #   room_type
    df = pd.get_dummies(df)

    # 正規化・正則化処理
    df_train, df_test, y_scaler = normalize(df)

    return df_train, df_test, y_scaler


if __name__ == "__main__":
    DATA_DIR = "./datasets/"
    train_data = pd.read_csv(DATA_DIR + "train.csv")
    test_data = pd.read_csv(DATA_DIR + "test.csv")
    train_data, test_data, y_scaler = preprocess(train_data, test_data)
