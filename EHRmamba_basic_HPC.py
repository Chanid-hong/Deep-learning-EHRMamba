import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning import Trainer  # Add this
from sklearn.model_selection import train_test_split  # Add this
from sklearn.metrics import accuracy_score, roc_auc_score,precision_score,recall_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt


# Function to preprocess the dataset
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


# Define the MortalityDataset class
class MortalityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        return patient['variables'], patient['time_stamps'], patient['values'], patient['label']


# Custom collate function
def collate_fn(batch):
    variables = [item[0] for item in batch]
    time_stamps = [item[1] for item in batch]
    values = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float)
    variables_padded = pad_sequence(variables, batch_first=True, padding_value=0)
    time_stamps_padded = pad_sequence(time_stamps, batch_first=True, padding_value=0)
    values_padded = pad_sequence(values, batch_first=True, padding_value=0)
    return variables_padded, time_stamps_padded, values_padded, labels

class MambaMortalityRisk(pl.LightningModule):
    def __init__(
            self,
            vocab_size=37,
            embedding_size=768,
            time_embeddings_size=32,
            num_hidden_layers=12,
            learning_rate=5e-5,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        # Embedding layers
        self.variable_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.time_embeddings = nn.Embedding(100, time_embeddings_size)
        self.time_project = nn.Linear(time_embeddings_size, embedding_size)
        self.value_layer = nn.Linear(1, embedding_size)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=8,  # Number of attention heads
            dim_feedforward=2048,  # Feedforward network size
            dropout=0.1,
        )
        self.backbone = TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )

    def test_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)

        # Compute the loss
        loss = nn.BCELoss()(predictions.squeeze(), labels.float())
        self.log("test_loss", loss, prog_bar=True)

        # Calculate metrics
        predicted_labels = (predictions.squeeze() > 0.5).float()
        precision = precision_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)

        # Log metrics
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        return {"test_loss": loss}

    def forward(self, variables, time_stamps, values):
        # Variable embeddings
        variable_embeds = self.variable_embeddings(variables)  # [batch_size, num_variables, embedding_size]
        variable_embeds = variable_embeds.unsqueeze(1).repeat(1, time_stamps.shape[1], 1, 1).mean(
            dim=2)  # [batch_size, time_steps, embedding_size]

        # Time embeddings
        time_embeds = self.time_embeddings(time_stamps.long())  # [batch_size, time_steps, time_embeddings_size]
        time_embeds = self.time_project(time_embeds)  # [batch_size, time_steps, embedding_size]

        # Value embeddings
        value_embeds = self.value_layer(values.unsqueeze(-1))  # [batch_size, time_steps, num_variables, embedding_size]
        value_embeds = value_embeds.mean(dim=2)  # [batch_size, time_steps, embedding_size]

        # Combine embeddings
        inputs = variable_embeds + time_embeds + value_embeds  # [batch_size, time_steps, embedding_size]
        inputs = inputs.permute(1, 0, 2)  # [seq_len, batch_size, embedding_size]

        # Transformer Encoder
        outputs = self.backbone(inputs)  # [seq_len, batch_size, embedding_size]

        # Classification Head
        pooled_output = outputs.mean(dim=0)  # Aggregate over the sequence dimension
        logits = self.classification_head(pooled_output)  # [batch_size, 1]
        return logits

    def training_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = nn.BCELoss()(predictions.squeeze(), labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = nn.BCELoss()(predictions.squeeze(), labels.float())
        self.log("val_loss", loss)

        predicted_labels = (predictions.squeeze() > 0.5).float()
        acc = (predicted_labels == labels).float().mean()
        self.log("val_accuracy", acc)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

# Initialize metrics collection
all_results = []

# rsync -avzh /Users/hanchunxu/Desktop/deep_learning/data/ s220311@transfer.gbar.dtu.dk:~/deep_learning_project/data/
# Define the root folder and split folders
root_folder = "/zhome/fb/b/174167/deep_learning_project/data/"
results_path = "/zhome/fb/b/174167/deep_learning_project/results_basic/"
os.makedirs(results_path, exist_ok=True)
split_folders = [f"split_{i}" for i in range(1, 6)]

for split_idx, split_folder in enumerate(split_folders, start=1):
    print(f"Processing {split_folder}...")
    split_path = os.path.join(root_folder, split_folder)

    # Load the data
    train_file = os.path.join(split_path, f"train_physionet2012_{split_idx}.npy")
    val_file = os.path.join(split_path, f"validation_physionet2012_{split_idx}.npy")
    test_file = os.path.join(split_path, f"test_physionet2012_{split_idx}.npy")
    train_data = preprocess_dataset(np.load(train_file, allow_pickle=True))
    val_data = preprocess_dataset(np.load(val_file, allow_pickle=True))
    test_data = preprocess_dataset(np.load(test_file, allow_pickle=True))

    # Create DataLoaders
    train_loader = DataLoader(MortalityDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MortalityDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(MortalityDataset(test_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the model
    model = MambaMortalityRisk(
        vocab_size=37,
        embedding_size=768,
        time_embeddings_size=32,
        num_hidden_layers=12,
        learning_rate=5e-5,
    )

    # Metrics tracking for loss curves
    train_losses = []
    val_losses = []


    class LossLoggerCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics["train_loss"].item()
            train_losses.append(train_loss)

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics["val_loss"].item()
            val_losses.append(val_loss)


    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{split_folder}",
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    loss_logger = LossLoggerCallback()

    # Trainer setup
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping, loss_logger],
    )

    # Train the model
    print(f"Training on {split_folder}...")
    trainer.fit(model, train_loader, val_loader)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    model = MambaMortalityRisk.load_from_checkpoint(best_model_path)

    # Evaluate the model on the test set
    print(f"Evaluating on {split_folder} test set...")
    # Ensure the model is on GPU
    device = torch.device("cuda")  # Use the GPU
    model = model.to(device)  # Move the model to GPU

    all_predictions, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move data to GPU
            variables, time_stamps, values, labels = batch
            variables = variables.to(device)
            time_stamps = time_stamps.to(device)
            values = values.to(device)
            labels = labels.to(device)
            predictions = model(variables, time_stamps, values).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, (all_predictions > 0.5))
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, (all_predictions > 0.5))
    recall = recall_score(all_labels, (all_predictions > 0.5))
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Collect metrics
    all_results.append({
        "Split": split_folder,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    })

    # Generate and save visualizations
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

# Save all results as a DataFrame
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(results_path, "metrics_summary.csv"), index=False)
print("All results saved to results/metrics_summary.csv.")