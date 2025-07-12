import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from balanced_loss import Loss
from copy import deepcopy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def token_accuracy(preds: torch.Tensor, targets: torch.Tensor, inverse_mask: torch.Tensor):
    preds_masked = preds.argmax(-1).masked_select(~inverse_mask)
    targets_masked = targets.masked_select(~inverse_mask)
    correct = (preds_masked == targets_masked).sum()
    return float(correct / preds_masked.size(0))

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, starting from epoch {epoch}")
    return epoch

class BertTrainer:
    def __init__(self, model, dataset, log_dir, checkpoint_path=None, batch_size=24, lr=0.005, epochs=5,
                 print_every=10, accuracy_every=50, early_stopping_patience=None, min_delta=0.0):
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.loss_fn = nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._print_every = print_every
        self._accuracy_every = accuracy_every
        self._batched_len = len(self.loader)

    def __call__(self, X_vocab, adj):
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss = self.train_one_epoch(epoch, X_vocab, adj)
            if self.checkpoint_path:
                save_checkpoint(self.model, self.optimizer, epoch, f"{self.checkpoint_path}/bert_mlm_epoch_{epoch}_{int(datetime.now().timestamp())}.pt")

            if self.early_stopping_patience is not None:
                if epoch_loss < best_loss - self.min_delta:
                    best_loss = epoch_loss
                    best_state = deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def train_one_epoch(self, epoch, X_vocab, adj):
        self.model.train()
        start_time = time.time()
        running_loss = 0
        total_loss = 0

        for batch_idx, (inp, mask, inv_mask, target) in tqdm(enumerate(self.loader, start=1), total=self._batched_len, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
            self.optimizer.zero_grad()

            output = self.model(inp, mask, X_vocab, adj)
            masked_output = output.masked_fill(inv_mask.unsqueeze(-1).expand_as(output), 0)

            mlm_loss = self.loss_fn(masked_output.transpose(1, 2), target)
            mlm_loss.backward()
            self.optimizer.step()

            running_loss += mlm_loss.item()
            total_loss += mlm_loss.item()

            if batch_idx % self._print_every == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / self._print_every
                print(f"[Epoch {epoch+1}/{self.epochs}] [{batch_idx}/{self._batched_len}] "
                      f"MLM Loss: {avg_loss:.4f} | Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                running_loss = 0

                if batch_idx % self._accuracy_every == 0:
                    acc = token_accuracy(output, target, inv_mask)
                    print(f"Token Accuracy: {acc:.4f}")
        return total_loss / self._batched_len
class BertTrainerClassification:
    def __init__(self, model, train_ds, test_ds, batch_size=24, lr=0.005, epochs=5, print_every=10,
                 checkpoint_path=None, early_stopping_patience=None, min_delta=0.0):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.loss_fn = Loss(
            loss_type="cross_entropy",
            samples_per_class=train_ds.class_weight(),
            class_balanced=True
        ).to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 50, 60], gamma=0.5)

        self._print_every = print_every

    def __call__(self, X_vocab, adj):
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss = self.train_one_epoch(epoch, X_vocab, adj)
            if self.checkpoint_path:
                save_checkpoint(self.model, self.optimizer, epoch, f"{self.checkpoint_path}/bert_classify_epoch_{epoch}_{int(datetime.now().timestamp())}.pt")
            self.evaluate(X_vocab, adj)
            self.scheduler.step()

            if self.early_stopping_patience is not None:
                if epoch_loss < best_loss - self.min_delta:
                    best_loss = epoch_loss
                    best_state = deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def train_one_epoch(self, epoch, X_vocab, adj):
        self.model.train()
        running_loss = 0
        total_loss = 0

        for batch_idx, (inp, mask, labels) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            logits = self.model(inp, mask, X_vocab, adj)
            loss = self.loss_fn(logits, labels.flatten())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_idx % self._print_every == 0:
                avg_loss = running_loss / self._print_every
                print(f"[Epoch {epoch+1}/{self.epochs}] [Batch {batch_idx}] Classification Loss: {avg_loss:.4f}")
                running_loss = 0
        return total_loss / len(self.train_loader)

    def evaluate(self, X_vocab, adj):
        self.model.eval()
        top1 = top3 = top5 = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inp, mask, label in self.test_loader:
                logits = self.model(inp, mask, X_vocab, adj)
                top_preds = torch.topk(logits, k=5).indices.squeeze(0)

                y_true.append(label.item())
                y_pred.append(top_preds[0].item())

                top1 += int(label in top_preds[:1])
                top3 += int(label in top_preds[:3])
                top5 += int(label in top_preds[:5])

        total = len(self.test_loader)
        result = {
            "acc@1": top1/total,
            "acc@3": top3/total,
            "acc@5": top5/total,
            # (macro) evaluations
            "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='macro', zero_division=0)
        }
        self.model.train()
        return result
