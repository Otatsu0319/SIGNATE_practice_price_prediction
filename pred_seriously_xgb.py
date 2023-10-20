import numpy as np
import pandas as pd
import xgboost as xgb

from preprocessor import drop_columns, text_to_num

# from sklearn.model_selection import train_test_split


# Load data
def load_data(path="./datasets"):
    train = pd.read_csv(path + "/train.csv")
    test = pd.read_csv(path + "/test.csv")
    return train, test


def main():
    train, test = load_data()
    train, test = drop_columns(train, test)
    train = text_to_num(train)
    test = text_to_num(test)

    df = pd.concat([train, test], axis=0, ignore_index=True)
    # 次の項目をone-hotエンコーディングする
    #   bed_type
    #   city
    #   cancellation_policy
    #   property_type
    #   room_type
    df = pd.get_dummies(df)
    train = df[df['y'].notna()]
    test = df[df['y'].isna()]

    train = train.astype(np.float32)
    ans = test.drop(['y'], axis=1).astype(np.float32)

    x, y = train.drop(["y"], axis=1), train["y"]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3020)
    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test, label=y_test)

    dtrain = xgb.DMatrix(x, label=y)
    dtest = xgb.DMatrix(ans)

    xgb_params = {
        # 二値分類
        'objective': 'reg:squarederror',
        # 評価指標loglossを仕様
        'eval_metric': 'rmse',
    }

    bst = xgb.train(
        xgb_params,
        dtrain,
        # 学習ラウンド数
        num_boost_round=1000,
        # 一定ラウンドを回しても改善が見込めない場合は学習を打ち切る
        # early_stopping_rounds=10,
    )

    y_pred = bst.predict(dtest)

    submit_file = pd.DataFrame({"id": test.index - test.index[0], 'y': y_pred})  # 提出用ファイルの作成
    # submit_file = pd.DataFrame({"id": test_data.index - test_data.index[0], 'y': average_ans})  # 提出用ファイルの作成
    submit_file.to_csv("submit.csv", index=False, header=False)  # csvファイルとして出力


if __name__ == "__main__":
    main()
