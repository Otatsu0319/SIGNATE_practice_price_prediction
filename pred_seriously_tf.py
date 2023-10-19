import numpy as np  # 計算処理などに活用
import pandas as pd  # CSVの読み込みと前処理などに活用
from preprocessor import preprocess
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow_addons as tfa
from models import build_model_regressor, build_model_regressor_v2
import tqdm
import collections


pd.set_option('display.max_columns', 200)  # 通常だと列が省略されるので拡張
DATA_DIR = "./datasets/"
K = 5


def load_datasets():
    train_data = pd.read_csv(DATA_DIR + "train.csv")
    test_data = pd.read_csv(DATA_DIR + "test.csv")  # ここでのtestは提出用に使うデータ(=正解の宿泊価格が入っていない)なので注意
    return preprocess(train_data, test_data)


class Regressor:
    def __init__(self):
        self.model = build_model_regressor()
        # self.model = build_model_regressor_v2()

        self.train_data, self.test_data, self.y_scaler = load_datasets()
        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)

        self.iteration = 1000
        self.patience = 1000
        self.moving_average_queue = collections.deque(maxlen=50)
        self.batch_size = 4096

        self.loss_obj_mse = tf.losses.MeanSquaredError()
        self.score_obj_rmse = tf.metrics.RootMeanSquaredError()
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tfa.optimizers.RectifiedAdam(
                learning_rate=0.0005, total_steps=self.iteration, warmup_proportion=0.1, min_lr=0.00001
            )
        )
        self.train_loss = tf.metrics.Mean("train_loss")

    def _summary_write(self, iteration):
        pass

    @tf.function(jit_compile=True)
    def train_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x, training=True)

            loss = self.loss_obj_mse(y, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def dev_test_step(self, x):
        y_pred = self.model(x, training=False)
        return y_pred

    def train_index(self, train_index, dev_index, suffix):
        self.summary_writer = tf.summary.create_file_writer(logdir=f"./logs/{suffix}")
        x_train = self.train_data.iloc[train_index].drop(['y'], axis=1)
        y_train = self.train_data.iloc[train_index]['y']
        x_dev = self.train_data.iloc[dev_index].drop(['y'], axis=1)
        y_dev = self.train_data.iloc[dev_index]['y']

        wait = 0
        best = 100.0

        train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache()
        train_set = (
            train_set.shuffle(train_set.cardinality())
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        dev_set = (
            tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
            .cache()
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        for i in tqdm.tqdm(range(self.iteration)):
            for x, y in tqdm.tqdm(train_set, leave=False):
                loss = self.train_step(x, y)

            self.train_loss.update_state(loss)
            loss = self.train_loss.result()
            self.moving_average_queue.append(loss)
            self.train_loss.reset_states()

            score = self.predict_score(dev_set)

            with self.summary_writer.as_default():
                tf.summary.scalar("train/loss", loss, step=i)
                tf.summary.scalar("train/score", score, step=i)

            wait += 1
            loss = np.mean(self.moving_average_queue)
            if loss < best:
                best = loss
                wait = 0
            if wait >= self.patience:
                break

        print(f"dev score: {self.y_scaler.inverse_transform(score[np.newaxis, np.newaxis]).squeeze()}")
        return self.model

    def train(self):
        self.summary_writer = tf.summary.create_file_writer(logdir="./logs/")
        x_train = self.train_data.drop(['y'], axis=1)
        y_train = self.train_data['y']

        wait = 0
        best = 0

        train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache()
        train_set = (
            train_set.shuffle(train_set.cardinality())
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        for i in tqdm.tqdm(range(self.iteration)):
            for x, y in tqdm.tqdm(train_set, leave=False):
                loss = self.train_step(x, y)
                self.train_loss.update_state(loss)
            loss = self.train_loss.result()
            with self.summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=i)
            self.train_loss.reset_states()

            wait += 1
            if loss < best:
                best = loss
                wait = 0
            if wait >= self.patience:
                break

        print(
            f"train score: {self.y_scaler.inverse_transform(self.predict_score(train_set, leave=True)[np.newaxis, np.newaxis]).squeeze()}"
        )
        return self.model

    def predict_score(self, data_set, leave=False):
        for x, y in tqdm.tqdm(data_set, leave=leave):
            y_pred = self.dev_test_step(x)
            self.score_obj_rmse.update_state(y, y_pred)
        score = self.score_obj_rmse.result()
        self.score_obj_rmse.reset_states()
        return score


def main():
    # データの読み込み

    # モデルの構築
    regressor = Regressor()

    # models = []
    # K分割交差法で学習
    kf = KFold(n_splits=K, shuffle=True, random_state=2013)
    for i, (train_index, dev_index) in enumerate(kf.split(regressor.train_data)):
        print(f"fold {i+1}")
        model = regressor.train_index(train_index, dev_index, f"fold_{i+1}")
        model.save(f"./models/model_{i+1}")
        regressor = Regressor()
        # break

    # 提出用ファイルの作成
    model = regressor.train()
    # model = regressor.model

    x_test = regressor.test_data.drop(['y'], axis=1)
    y_test_pred = model.predict(x_test)  # 予測実施
    y_test_pred = regressor.y_scaler.inverse_transform(y_test_pred[:, np.newaxis]).squeeze()  # 正規化を元に戻す
    submit_file = pd.DataFrame(
        {"id": regressor.test_data.index - regressor.test_data.index[0], 'y': y_test_pred.squeeze()}
    )  # 提出用ファイルの作成
    submit_file.to_csv("submit.csv", index=False, header=False)  # csvファイルとして出力


def average_prediction():
    train_data, test_data, y_scaler = load_datasets()
    x_test = test_data.drop(['y'], axis=1)
    average_ans = np.zeros(len(x_test))
    x_test = x_test.astype(np.float32)

    for i in range(K):
        model = tf.keras.models.load_model(f"./models/model_{i+1}")

        y_test_pred = model.predict(x_test)  # 予測実施
        y_test_pred = y_scaler.inverse_transform(y_test_pred).squeeze()  # 正規化を元に戻す
        average_ans += y_test_pred

    average_ans /= 5

    submit_file = pd.DataFrame({"id": test_data.index - test_data.index[0], 'y': average_ans})  # 提出用ファイルの作成
    submit_file.to_csv("submit.csv", index=False, header=False)  # csvファイルとして出力


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) > 0:
        for k in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[k], True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15155)]
    # )  # Notice here

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    # main()
    average_prediction()
