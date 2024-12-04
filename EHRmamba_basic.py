from typing import Any, Dict, Tuple, List, Optional, Union  # Ensure typing utilities are imported

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning import Trainer  # Add this
from sklearn.model_selection import train_test_split  # Add this
from sklearn.metrics import accuracy_score, roc_auc_score,precision_score,recall_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt

def preprocess_dataset(test):
    dataset = []  # Initialize list to store patient dictionaries

    for row_index in range(test.shape[0]):
        row_data = test[row_index]

        # Extract data
        ts_values = row_data['ts_values']  # Time-series values (2D array)
        ts_times = row_data['ts_times']  # Time indices
        labels = row_data['labels']  # Mortality label (0 or 1)

        # Prepare the patient dictionary
        patient_dict = {
            'variables': torch.tensor(range(ts_values.shape[1]), dtype=torch.long),
            'time_stamps': torch.tensor(ts_times, dtype=torch.float),
            'values': torch.tensor(ts_values, dtype=torch.float),
            'label': labels  # Mortality label
        }

        # Append to the dataset
        dataset.append(patient_dict)

    return dataset

class MortalityDataset(Dataset):
    """Dataset class for mortality risk prediction."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        return patient['variables'], patient['time_stamps'], patient['values'], patient['label']

def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    Args:
        batch: List of tuples (variables, time_stamps, values, label)
    Returns:
        Padded tensors for variables, time_stamps, values, and labels.
    """
    variables = [item[0][:MAX_SEQ_LENGTH] for item in batch]  # Truncate sequences if needed
    time_stamps = [item[1][:MAX_SEQ_LENGTH] for item in batch]
    values = [item[2][:MAX_SEQ_LENGTH] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float)

    # Pad sequences to the maximum length in the batch
    variables_padded = pad_sequence(variables, batch_first=True, padding_value=0)
    time_stamps_padded = pad_sequence(time_stamps, batch_first=True, padding_value=0)
    values_padded = pad_sequence(values, batch_first=True, padding_value=0)

    return variables_padded, time_stamps_padded, values_padded, labels\

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


# Load the .npy file and preprocess the dataset
test = np.load('/Users/hanchunxu/Desktop/deep_learning/P12data/split_1/test_physionet2012_1.npy', allow_pickle=True)
processed_dataset = preprocess_dataset(test)

MAX_SEQ_LENGTH = 100  # Adjust this based on the dataset characteristics

# Split the dataset
train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create DataLoaders with the custom collate function
train_loader = DataLoader(
    MortalityDataset(train_data),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    MortalityDataset(val_data),
    batch_size=32,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    MortalityDataset(test_data),
    batch_size=32,
    collate_fn=collate_fn
)

# Initialize the model
model = MambaMortalityRisk(
    vocab_size=37,
    embedding_size=768,
    time_embeddings_size=32,
    num_hidden_layers=12,
    learning_rate=5e-5,
)

trainer = Trainer(
    max_epochs=10,
    accelerator="cpu",
    devices=1,
    log_every_n_steps=10,
)
trainer.fit(model, train_loader, val_loader)

# Collect predictions and true labels
all_predictions = []
all_labels = []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for batch in test_loader:
        variables, time_stamps, values, labels = batch
        predictions = model(variables, time_stamps, values).squeeze()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
all_predictions = torch.tensor(all_predictions)
all_labels = torch.tensor(all_labels)
accuracy = accuracy_score(all_labels.numpy(), (all_predictions > 0.5).numpy())
roc_auc = roc_auc_score(all_labels.numpy(), all_predictions.numpy())

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

# Plot ROC Curve

plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(all_labels.numpy(), all_predictions.numpy())
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save the figure
output_path = "roc_curve.png"  # Define the output path
plt.savefig(output_path, dpi=300)  # Save with high resolution
plt.close()

# Compute confusion matrix
binary_predictions = (all_predictions > 0.5).int()  # Convert probabilities to binary predictions
cm = confusion_matrix(all_labels, binary_predictions.cpu().numpy())

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Print confusion matrix values
print("Confusion Matrix:")
print(cm)

# Optionally calculate additional metrics from the confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Calculate and display rates
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")