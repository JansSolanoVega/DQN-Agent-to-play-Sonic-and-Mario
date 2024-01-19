from torch.utils.tensorboard import SummaryWriter
import os
class DataLogger:
    def __init__(self, model="DDQN"):
        folder_path = os.path.join("runs", model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.writer = SummaryWriter(folder_path)
        self.loss_accum = 0
    
    def add(self, loss, step):
        self.loss_accum += loss
        self.writer.add_scalar('training loss', float(self.loss_accum), step)
    
    def close(self):
        self.writer.close()