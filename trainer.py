import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DEVICE

def train_model(model, train_dataset, val_dataset, criterion, optimizer, epochs):
    best_val = float("inf")
    history = {"train": [], "val": []}

    dl_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in dl_train:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in dl_val:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(Xb), yb).item())

        tr, va = np.mean(train_losses), np.mean(val_losses)
        history["train"].append(tr)
        history["val"].append(va)
        print(f"Epoch {ep:02d} Train {tr:.6f} Val {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), "best_model.pth")

    plt.figure(figsize=(8,5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.title("Smooth L1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

    return history

def evaluate_model(model, test_dataset, criterion):
    from torch.utils.data import DataLoader
    dl_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    losses = []
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    with torch.no_grad():
        for Xb, yb in dl_test:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            losses.append(criterion(model(Xb), yb).item())
    return np.mean(losses)

def predict_next(model, recent_features, feat_scaler, tar_scaler):
    import torch
    model.eval()
    seq = torch.tensor(recent_features[np.newaxis,...], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(seq).cpu().numpy()
    return tar_scaler.inverse_transform(pred_norm).flatten()
