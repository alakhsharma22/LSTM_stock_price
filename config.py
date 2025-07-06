import torch
TICKER     = "AAPL"
START_DATE = "2021-07-06"
END_DATE   = "2025-07-06"
SEQ_LEN    = 60
BATCH_SIZE = 32
HIDDEN_SZ  = 50
NUM_LAYERS = 2
DROPOUT    = 0.2
LR         = 1e-3
EPOCHS     = 50
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
