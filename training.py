# training.py

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from balanced_loss import Loss  # Ensure this exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def token_accuracy(preds: torch.Tensor, targets: torch.Tensor, inverse_mask: torch.Tensor):
    preds_masked = preds.argmax(-1).masked_select(~inverse_mask)
    targets_masked = targets.masked_select(~inverse_mask)
    correct = (preds_masked == targets_masked).sum()
    return float(correct / preds_masked.size(0))


class BertTrainer:
    def __init__(self, model, dataset, log_dir, checkpoint_path=None, batch_size=24, lr=0.005, epochs=5,
                 print_every=10, accuracy_every=50):
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.loss_fn = nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._print_every = print_every
        self._accuracy_every = accuracy_every
        self._batched_len = len(self.loader)

    def __call__(self, X_vocab, adj):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch, X_vocab, adj)
            self.save_checkpoint(epoch)

    def train_one_epoch(self, epoch, X_vocab, adj):
        self.model.train()
        start_time = time.time()
        running_loss = 0

        for batch_idx, (inp, mask, inv_mask, target) in enumerate(self.loader, start=1):
            self.optimizer.zero_grad()

            output = self.model(inp, mask, X_vocab, adj)
            masked_output = output.masked_fill(inv_mask.unsqueeze(-1).expand_as(output), 0)

            mlm_loss = self.loss_fn(masked_output.transpose(1, 2), target)
            mlm_loss.backward()
            self.optimizer.step()

            running_loss += mlm_loss.item()

            if batch_idx % self._print_every == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / self._print_every
                print(f"[Epoch {epoch+1}/{self.epochs}] [{batch_idx}/{self._batched_len}] "
                      f"MLM Loss: {avg_loss:.4f} | Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                running_loss = 0

                if batch_idx % self._accuracy_every == 0:
                    acc = token_accuracy(output, target, inv_mask)
                    print(f"Token Accuracy: {acc:.4f}")

    def save_checkpoint(self, epoch):
        if not self.checkpoint_path:
            return
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        filename = f"{self.checkpoint_path}/bert_checkpoint_epoch{epoch}_{int(datetime.utcnow().timestamp())}.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")


class BertTrainerClassification:
    def __init__(self, model, train_ds, test_ds, batch_size=24, lr=0.005, epochs=5, print_every=10):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        self.epochs = epochs

        self.loss_fn = Loss(
            loss_type="cross_entropy",
            samples_per_class=train_ds.class_weight(),
            class_balanced=True
        ).to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 50, 60], gamma=0.5)

        self._print_every = print_every

    def __call__(self, X_vocab, adj):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch, X_vocab, adj)
            self.evaluate(X_vocab, adj)
            self.scheduler.step()

    def train_one_epoch(self, epoch, X_vocab, adj):
        self.model.train()
        running_loss = 0

        for batch_idx, (inp, mask, labels) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            logits = self.model(inp, mask, X_vocab, adj)
            loss = self.loss_fn(logits, labels.flatten())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % self._print_every == 0:
                avg_loss = running_loss / self._print_every
                print(f"[Epoch {epoch+1}/{self.epochs}] [Batch {batch_idx}] Classification Loss: {avg_loss:.4f}")
                running_loss = 0

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
        print(f"Top@1: {top1/total:.4f} | Top@3: {top3/total:.4f} | Top@5: {top5/total:.4f}")
        print("F1 (macro):", f1_score(y_true, y_pred, average='macro'))
        print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
        print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
        self.model.train()
