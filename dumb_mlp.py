import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    X = np.column_stack(tuple([i for i in sr_data.iloc[:, 2:].values])).transpose()

    Y = np.column_stack(tuple(interp_list)) 
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=700,
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
    plt.savefig("dumb_mlp_metrics.png")


if __name__ == "__main__":
    main()
