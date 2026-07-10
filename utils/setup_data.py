import pandas as pd
from numpy import interp
import torch


@torch.no_grad()
def setup_data(mtp: str, srp: str, steps: int, split: float, device):
    mt_data = pd.read_csv(mtp).dropna()
    sr_data = pd.read_csv(srp, skiprows=6).dropna()

    for i in range(1, 5):
        mt_data.iloc[:, i] -= mt_data.iloc[0, i]
    for i in range(2, 11):
        sr_data.iloc[:, i] -= sr_data.iloc[0, i]

    srnt = torch.from_numpy(sr_data.copy().iloc[:, 2:].values.copy()).to(device=device, dtype=torch.float32)

    mt = torch.column_stack(
        [
            torch.from_numpy(
                interp(
                    sr_data["Time"].values,
                    mt_data["Time"].values,
                    mt_data.iloc[:, i + 1].values,
                )
            ).to(torch.float32)
            for i in range(4)
        ]
    ). to(device=device, dtype=torch.float32)
    assert srnt.shape[0] == mt.shape[0]

    split_idx = int(srnt.shape[0] * split)
    train_Y, val_Y = torch.split(mt, [split_idx, mt.shape[0] - split_idx], dim=0)
    train_srnt, val_srnt = torch.split(
        srnt, [split_idx, srnt.shape[0] - split_idx], dim=0
    )

    y_mean = train_Y.mean(0, keepdim=True)
    y_std = train_Y.std(0, keepdim=True)

    train_Y = (train_Y - y_mean) / y_std
    val_Y = (val_Y - y_mean) / y_std

    srnt_mean = train_srnt.mean(dim=0, keepdim=True)
    srnt_std = train_srnt.std(dim=0, keepdim=True)
    train_srnt = (train_srnt - srnt_mean) / srnt_std
    val_srnt = (val_srnt - srnt_mean) / srnt_std

    train_Y, val_Y = train_Y[steps:], val_Y[steps:]

    train_X = torch.hstack(
        [train_srnt[steps - i : len(train_srnt) - i] for i in range(steps + 1)]
    )
    val_X = torch.hstack(
        [val_srnt[steps - i : len(val_srnt) - i] for i in range(steps + 1)]
    )

    train_X = train_X.view(train_X.shape[0], steps + 1, -1).flip(1)
    val_X = val_X.view(val_X.shape[0], steps + 1, -1).flip(1)

    return (train_X, train_Y, val_X, val_Y, y_mean, y_std, srnt_mean, srnt_std)
