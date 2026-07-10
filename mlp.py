import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAGS = 3
# TODO make  a dumb xyz=>motor vals

def setup_data(mt_path, sr_path):
    mt_data = pd.read_csv(mt_path)
    sr_data = pd.read_csv(sr_path, skiprows=6)

    mt_data = mt_data.dropna()
    sr_data = sr_data.dropna()

    for i in range(1, 5):
        mt_data.iloc[:, i] -= mt_data.iloc[0, i]
    for i in range(2, 11):
        sr_data.iloc[:, i] -= sr_data.iloc[0, i]

    return (mt_data, sr_data)


def rpy(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1

    z_axis = np.cross(v1, v2)
    z_unit = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)

    x_axis = p3 - (p1 + p2) / 2
    x_unit = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)

    y_unit = np.cross(z_unit, x_unit)
    rot_matrix = np.stack((x_unit, y_unit, z_unit), axis=-1)

    roll = np.arctan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])
    pitch = np.arctan2(
        -rot_matrix[:, 2, 0],
        np.sqrt(rot_matrix[:, 2, 1] ** 2 + rot_matrix[:, 2, 2] ** 2),
    )
    yaw = np.arctan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])

    return np.column_stack((roll, pitch, yaw))


def prevel_diff(x_raw, lags=3):
    t_prevs = np.array([x_raw[lags - i : len(x_raw) - i] for i in range(lags + 1)])
    for i in range(1, len(t_prevs)):
        t_prevs[i] = t_prevs[0] - t_prevs[i]
    return np.hstack(t_prevs)


def main():
    mt_data, sr_data = setup_data("motor_telemetry.csv", "softrobot.csv")

    interp_list = []
    for i in range(4):
        interp_list.append(
            np.interp(
                sr_data["Time"].values,
                mt_data["Time"].values,
                mt_data.iloc[:, i + 1].values,
            )
        )

    x_all = np.column_stack(tuple([i for i in sr_data.iloc[1:, 2:].values])).transpose()

    rpy_list = rpy(x_all[LAGS:, :3], x_all[LAGS:, 3:6], x_all[LAGS:, 6:9])
    one_diff = prevel_diff(x_all[:, :], LAGS)

    X = np.hstack((rpy_list, one_diff))
    Y = np.column_stack(tuple(interp_list))[
        LAGS + 1 :
    ]  # first point due to rpy not taken
    print(X.shape, Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=600,
        random_state=42,
        alpha=0.001,
        learning_rate_init=0.001,
        activation="relu",
        solver="adam",
    )
    model.fit(X_train_scaled, Y_train)

    Y_pred = model.predict(X_test_scaled)

    print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
    print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))
    print("R^2 Score:", r2_score(Y_test, Y_pred))

    fig, ax = plt.subplots(5, 1, figsize=(10, 15))

    ax[0].plot(Y_test[:, 0][:6000], label="Actual", color="green")
    ax[0].plot(Y_pred[:, 0][:6000], label="Prediction", color="purple")
    ax[0].set_title("motor_0")
    ax[1].plot(Y_test[:, 1][:6000], label="Actual", color="green")
    ax[1].plot(Y_pred[:, 1][:6000], label="Prediction", color="purple")
    ax[1].set_title("motor_1")
    ax[2].plot(Y_test[:, 2][:6000], label="Actual", color="green")
    ax[2].plot(Y_pred[:, 2][:6000], label="Prediction", color="purple")
    ax[2].set_title("motor_2")
    ax[3].plot(Y_test[:, 3][:6000], label="Actual", color="green")
    ax[3].plot(Y_pred[:, 3][:6000], label="Prediction", color="purple")
    ax[3].set_title("motor_3")
    ax[4].plot(model.loss_curve_)
    ax[4].set_title("loss_curve")
    plt.savefig("mlp_metrics.png")


if __name__ == "__main__":
    main()
