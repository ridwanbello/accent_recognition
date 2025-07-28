class Config:
    DATA_PATH = "/content/data/accent_data"
    SAMPLE_RATE = 16000
    N_MFCC = 13
    MAX_LEN = 100  # Padding/truncating

    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DROPOUT = 0.3
    NUM_CLASSES = 4

    DEVICE = "cuda"  # or "cpu"
    KFOLDS = 5
    EARLY_STOPPING_PATIENCE = 5
