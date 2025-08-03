import torch
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        patience: how many epochs to wait after last improvement
        min_delta: minimum change to be considered an improvement
        mode: 'min' (e.g., for loss) or 'max' (e.g., for accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return

        if self.mode == 'min':
            improvement = self.best_score - metric
            if improvement > self.min_delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            improvement = metric - self.best_score
            if improvement > self.min_delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.
    """
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y-%m-%d") 
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)