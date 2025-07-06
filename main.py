from config import *
from data import prepare_data
from dataset import create_sequences, StockDataset
from model import LSTMMultiTask
from trainer import train_model, evaluate_model, predict_next
from sklearn.model_selection import train_test_split

def main():
    feature_cols = [
        "Open","High","Low","Close","Volume",
        "MA10","MA20","BB_up","BB_low",
        "RSI14","MACD","MACD_signal",
        "Open_prev","OC_diff"
    ]
    target_cols = ["Open","High","Low","Close"]

    feats, tars, feat_scaler, tar_scaler = prepare_data(
        feature_cols, target_cols, TICKER, START_DATE, END_DATE
    )

    X, Y = create_sequences(feats, tars, SEQ_LEN)
    Xtr, Xtmp, Ytr, Ytmp = train_test_split(X, Y, test_size=0.30, shuffle=False)
    Xvl, Xte, Yvl, Yte = train_test_split(Xtmp, Ytmp, test_size=0.50, shuffle=False)

    train_ds = StockDataset(Xtr, Ytr)
    val_ds   = StockDataset(Xvl, Yvl)
    test_ds  = StockDataset(Xte, Yte)

    model     = LSTMMultiTask(len(feature_cols), HIDDEN_SZ, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = __import__('torch').nn.SmoothL1Loss()
    optimizer = __import__('torch').optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    train_model(model, train_ds, val_ds, criterion, optimizer, EPOCHS)

    test_loss = evaluate_model(model, test_ds, criterion)
    print("Test Loss:", test_loss)

    recent_feats = feats[-SEQ_LEN:]
    pred = predict_next(model, recent_feats, feat_scaler, tar_scaler)
    print(f"Next-day OPEN:  ${pred[0]:.2f}")
    print(f"Next-day HIGH:  ${pred[1]:.2f}")
    print(f"Next-day LOW:   ${pred[2]:.2f}")
    print(f"Next-day CLOSE: ${pred[3]:.2f}")

if __name__ == "__main__":
    main()
