from utils.setup_data import setup_data
from utils.model import LSTMStack
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.manual_seed(42)

SR_PATH = "softrobot.csv"
MT_PATH = "motor_telemetry.csv"
TIME_STEPS = 5
DATA_SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
WD = 1e-4

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_Y, val_X, val_Y, y_mean, y_std, x_mean, x_std = setup_data(
        MT_PATH, SR_PATH, TIME_STEPS, DATA_SPLIT, device=device
    )

    train_dataset = TensorDataset(train_X, train_Y)
    val_dataset = TensorDataset(val_X, val_Y)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    model = LSTMStack(train_X.shape[2], train_Y.shape[1])
    model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=True)

    eval_ps = []
    train_ps = []
    pbar = tqdm(range(EPOCHS))
    for i in pbar:
        train_loss = 0.0
        eval_loss = 0.0
        model.train()

        for batch_inp, batch_trg in train_loader:
            batch_inp = batch_inp.to(device)
            batch_trg = batch_trg.to(device)

            batch_pred, batch_loss = model(batch_inp, batch_trg)
            train_loss += batch_loss.item()

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss /= len(train_loader)
        train_ps.append(train_loss)

        model.eval()

        with torch.no_grad():
            for batch_inp, batch_trg in val_loader:
                batch_inp = batch_inp.to(device)
                batch_trg = batch_trg.to(device)

                pred, loss = model(batch_inp, batch_trg)

                eval_loss += loss.item()
        eval_loss /= len(val_loader)
        eval_ps.append(eval_loss)

        pbar.set_description(
            f"epoch {i}/{EPOCHS} "
            f"train loss mean: {train_loss:.4f} "
            f"eval loss mean: {eval_loss:.4f}"
        )

fig, ax = plt.subplots( 2, 1, figsize=(10, 10))
ax[0].plot(train_ps, label="Train Loss", color="blue")
ax[0].set_title("Train Loss")
ax[1].plot(eval_ps, label="Eval Loss", color="orange")
ax[1].set_title("Eval Loss")
plt.savefig("lstm_metrics.png")

sample = torch.tensor([
    [-1.496943,1.380387,0.985372,-1.516569,1.390089,0.962106,-1.539833,1.391180,0.982984],
    [-1.497029,1.380108,0.987315,-1.516680,1.390014,0.964157,-1.539924,1.390869,0.985093],
    [-1.497098,1.379786,0.989300,-1.516729,1.389877,0.966173,-1.539925,1.390532,0.987241],
    [-1.497113,1.379449,0.991257,-1.516793,1.389731,0.968212,-1.539976,1.390233,0.989341],
    [-1.497207,1.379142,0.993176,-1.516865,1.389587,0.970171,-1.540041,1.389893,0.991333],
    [-1.497306,1.378811,0.994995,-1.516957,1.389456,0.972090,-1.540151,1.389536,0.993254],
    [-1.497454,1.378434,0.996742,-1.517126,1.389309,0.973944,-1.540371,1.389159,0.995049],
    [-1.497639,1.378123,0.998362,-1.517305,1.389132,0.975565,-1.540519,1.388828,0.996696],
    [-1.497830,1.377851,0.999789,-1.517479,1.389012,0.977053,-1.540704,1.388511,0.998159],
    [-1.498040,1.377633,1.001042,-1.517659,1.388872,0.978297,-1.540914,1.388271,0.999360],
    [-1.498233,1.377425,1.002075,-1.517803,1.388806,0.979362,-1.541095,1.388023,1.000431],
], dtype=torch.float32, device=device)


sample = (sample - x_mean) / x_std
sample = sample.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    pred = model(sample, None)
    pred = pred * y_std + y_mean
    print("Predicted motor values for a sample sequence:", pred.squeeze(0))

