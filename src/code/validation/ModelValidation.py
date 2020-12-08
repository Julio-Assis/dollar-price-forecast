from src.code.models.Model import Model
from sklearn.metrics import r2_score
import numpy as np

def train_test_split(df, train_split_size):
    total_rows, total_columns = df.shape

    train_size = round(total_rows * train_split_size)
    test_size = total_rows - train_size

    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]

    return train_df, test_df


def compute_adjusted_r2(r2, sample_size, variables_count):

  return 1 - (1. - r2) * (sample_size - 1) / (sample_size - variables_count - 1)


def scale_inverse(y, scaling_max, scaling_min):
  return y * (scaling_max - scaling_min) + scaling_min

def mean_absolute_percentage_error(y_true_df, y_pred, scaling_max, scaling_min):

  y_true = scale_inverse(y_true_df.values.reshape(-1), scaling_max, scaling_min)

  y_pred_work = scale_inverse(y_pred, scaling_max, scaling_min)
  # import ipdb; ipdb.set_trace()
  norm_diff = np.abs(y_true - y_pred_work) / y_true

  # import ipdb; ipdb.set_trace()
  return norm_diff.sum() * 100. / len(y_true)


class ModelValidation:

    def __init__(self, model: Model, X, y, scaling_max, scaling_min, initial_train_size=0.8):
        self.evaluating_model = model

        self.X = X.copy()
        self.y = y.copy()
        self.initial_train_size = initial_train_size
        self.scaling_max = scaling_max
        self.scaling_min = scaling_min


    def validate(self):
        X_train, X_test = train_test_split(self.X, self.initial_train_size)
        y_train, y_test = train_test_split(self.y, self.initial_train_size)

        self.evaluating_model.train(X_train, y_train)

        y_pred = np.zeros((y_test.shape[0],))

        print(f'There are N={len(y_test)} weeks of test')

        for i in range(len(y_test)):
            y_pred[i] = self.evaluating_model.predict(X_test.iloc[i:i+1, :])
            self.evaluating_model.train(X_test.iloc[i:i+1, :], y_test.iloc[i])

            if i % 10 == 9:
                r2 = r2_score(y_true=y_test.iloc[:i], y_pred=y_pred[:i])
                print(f'week={i}')
                print(f'partial r2={r2}')
                print(f'''mean absolute percentage error MAPE={mean_absolute_percentage_error(
                    y_true_df=y_test.iloc[:i],
                    y_pred=y_pred[:i],
                    scaling_max=self.scaling_max,
                    scaling_min=self.scaling_min)}''')
                print(f'partial adjusted r2={compute_adjusted_r2(r2, i + 1, X_train.shape[0])}')
                print('')

        final_r2 = r2_score(y_true=y_test, y_pred=y_pred)
        print(f'Final computed r2={final_r2}')
        print(f'Final computed adjusted r2={compute_adjusted_r2(final_r2, len(y_test), X_train.shape[0])}')
        print(f'''Final mean absolute percentage error MAPE={mean_absolute_percentage_error(
                    y_true_df=y_test,
                    y_pred=y_pred,
                    scaling_max=self.scaling_max,
                    scaling_min=self.scaling_min)}''')
        return y_test, y_pred
