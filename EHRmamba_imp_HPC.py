import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, f1_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd

# Define FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, labels):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(predictions.squeeze(), labels.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# Define ImprovedMambaMortalityRisk Model
class ImprovedMambaMortalityRisk(pl.LightningModule):
    def __init__(
        self,
        vocab_size=37,
        embedding_size=768,
        time_embeddings_size=32,
        num_hidden_layers=12,
        learning_rate=5e-5,
        positive_class_weight=5.0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.positive_class_weight = positive_class_weight
        self.loss_function = FocalLoss(alpha=self.positive_class_weight)

        self.variable_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.time_embeddings = nn.Embedding(100, time_embeddings_size)
        self.time_project = nn.Linear(time_embeddings_size, embedding_size)
        self.value_layer = nn.Linear(1, embedding_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.backbone = TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        self.classification_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, variables, time_stamps, values):
        variable_embeds = self.variable_embeddings(variables).unsqueeze(1).repeat(1, time_stamps.shape[1], 1, 1).mean(dim=2)
        time_embeds = self.time_project(self.time_embeddings(time_stamps.long()))
        value_embeds = self.value_layer(values.unsqueeze(-1)).mean(dim=2)
        inputs = variable_embeds + time_embeds + value_embeds
        inputs = inputs.permute(1, 0, 2)
        outputs = self.backbone(inputs)
        pooled_output = outputs.mean(dim=0)
        logits = self.classification_head(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.loss_function(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.loss_function(predictions, labels)
        self.log("val_loss", loss)

        predicted_labels = (predictions.squeeze() > 0.5).float()
        precision = precision_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        acc = accuracy_score(labels.cpu(), predicted_labels.cpu())

        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_accuracy", acc)
        return loss

    def test_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.loss_function(predictions, labels)
        self.log("test_loss", loss)

        predicted_labels = (predictions.squeeze() > 0.5).float()
        precision = precision_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        acc = accuracy_score(labels.cpu(), predicted_labels.cpu())

        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_accuracy", acc)

        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)


# Preprocessing and Dataset Classes
def preprocess_dataset(test):
    dataset = []
    for row_data in test:
        ts_values = row_data['ts_values']
        ts_times = row_data['ts_times']
        labels = row_data['labels']
        dataset.append({
            'variables': torch.tensor(range(ts_values.shape[1]), dtype=torch.long),
            'time_stamps': torch.tensor(ts_times, dtype=torch.float),
            'values': torch.tensor(ts_values, dtype=torch.float),
            'label': labels,
        })
    return dataset


class MortalityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        return patient['variables'], patient['time_stamps'], patient['values'], patient['label']


def collate_fn(batch):
    variables, time_stamps, values, labels = zip(*batch)
    variables = pad_sequence(variables, batch_first=True, padding_value=0)
    time_stamps = pad_sequence(time_stamps, batch_first=True, padding_value=0)
    values = pad_sequence(values, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return variables, time_stamps, values, labels


# Training and Evaluation Pipeline for Multiple Splits
root_folder = "/zhome/fb/b/174167/deep_learning_project/data/"
results_path = "/zhome/fb/b/174167/deep_learning_project/results_improved/"
os.makedirs(results_path, exist_ok=True)
split_folders = [f"split_{i}" for i in range(1, 6)]

all_results = []

for split_idx, split_folder in enumerate(split_folders, start=1):
    print(f"Processing {split_folder}...")
    split_path = os.path.join(root_folder, split_folder)

    train_file = os.path.join(split_path, f"train_physionet2012_{split_idx}.npy")
    val_file = os.path.join(split_path, f"validation_physionet2012_{split_idx}.npy")
    test_file = os.path.join(split_path, f"test_physionet2012_{split_idx}.npy")
    train_data = preprocess_dataset(np.load(train_file, allow_pickle=True))
    val_data = preprocess_dataset(np.load(val_file, allow_pickle=True))
    test_data = preprocess_dataset(np.load(test_file, allow_pickle=True))

    train_loader = DataLoader(MortalityDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MortalityDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(MortalityDataset(test_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

    train_losses, val_losses = [], []

    class LossLoggerCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics["train_loss"].item()
            train_losses.append(train_loss)

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics["val_loss"].item()
            val_losses.append(val_loss)

    model = ImprovedMambaMortalityRisk(
        vocab_size=37, embedding_size=768, time_embeddings_size=32,
        num_hidden_layers=12, learning_rate=5e-5, positive_class_weight=5.0
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{split_folder}",
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    loss_logger = LossLoggerCallback()

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping, loss_logger],
    )

    print(f"Training on {split_folder}...")
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    model = ImprovedMambaMortalityRisk.load_from_checkpoint(best_model_path)

    # Ensure the model is on GPU
    device = torch.device("cuda")  # Use the GPU
    model = model.to(device)  # Move the model to GPU

    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            variables, time_stamps, values, labels = batch
            variables, time_stamps, values, labels = variables.cuda(), time_stamps.cuda(), values.cuda(), labels.cuda()
            predictions = model(variables, time_stamps, values).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, (all_predictions > 0.5))
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, (all_predictions > 0.5))
    recall = recall_score(all_labels, (all_predictions > 0.5))
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    all_results.append({
        "Split": split_folder,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    })

    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, (all_predictions > 0.5))

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {split_folder}")
    plt.legend()
    plt.savefig(os.path.join(results_path, f"{split_folder}_roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {split_folder}")
    plt.legend()
    plt.savefig(os.path.join(results_path, f"{split_folder}_precision_recall_curve.png"))
    plt.close()

    # Loss Curve
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {split_folder}")
    plt.legend()
    plt.savefig(os.path.join(results_path, f"{split_folder}_loss_curve.png"))
    plt.close()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {split_folder}")
    plt.savefig(os.path.join(results_path, f"{split_folder}_confusion_matrix.png"))
    plt.close()

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(results_path, "metrics_summary.csv"), index=False)
print("All results saved.")