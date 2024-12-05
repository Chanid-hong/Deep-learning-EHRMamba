from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning import Trainer  # Add this
from sklearn.model_selection import train_test_split  # Add this
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, f1_score, classification_report
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class NewMambaMortalityRisk(pl.LightningModule):
    def __init__(
        self,
        vocab_size=37,
        embedding_size=768,
        time_embeddings_size=32,
        num_hidden_layers=12,
        learning_rate=5e-5,
        positive_class_weight=1.0,  # Add positive class weight
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.positive_class_weight = positive_class_weight

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
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.Dropout(0.3),  # Add another dropout
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, variables, time_stamps, values):
        # Variable embeddings
        variable_embeds = self.variable_embeddings(variables)  # [batch_size, num_variables, embedding_size]
        variable_embeds = variable_embeds.unsqueeze(1).repeat(1, time_stamps.shape[1], 1, 1).mean(dim=2)  # [batch_size, time_steps, embedding_size]

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

    def compute_loss(self, predictions, labels):
        # Weighted Binary Cross-Entropy Loss
        weights = torch.where(labels == 1, self.positive_class_weight, 1.0)
        loss = F.binary_cross_entropy(predictions.squeeze(), labels.float(), weight=weights)
        return loss

    def training_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.compute_loss(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.compute_loss(predictions, labels)
        self.log("val_loss", loss)

        # Calculate metrics
        predicted_labels = (predictions.squeeze() > 0.5).float()
        precision = precision_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        acc = accuracy_score(labels.cpu(), predicted_labels.cpu())

        # Log metrics
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_accuracy", acc)

        return loss

    def test_step(self, batch, batch_idx):
        variables, time_stamps, values, labels = batch
        predictions = self(variables, time_stamps, values)
        loss = self.compute_loss(predictions, labels)
        self.log("test_loss", loss)

        # Calculate metrics
        predicted_labels = (predictions.squeeze() > 0.5).float()
        precision = precision_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=1)
        acc = accuracy_score(labels.cpu(), predicted_labels.cpu())

        # Log metrics
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_accuracy", acc)

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

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

class MortalityDataset(Dataset):
    """Dataset class for mortality risk prediction."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        return patient['variables'], patient['time_stamps'], patient['values'], patient['label']

class UpsampledMortalityDataset(Dataset):
    """Dataset class that upsamples the minority class."""
    def __init__(self, data):
        """
        Args:
            data: List of dictionaries, where each dict contains
                  'variables', 'time_stamps', 'values', and 'label'.
        """
        self.data = data

        # Separate data into positive and negative classes
        self.negative_samples = [item for item in data if item['label'] == 0]
        self.positive_samples = [item for item in data if item['label'] == 1]

        # Calculate the upsampling factor
        if len(self.positive_samples) == 0:
            raise ValueError("No positive samples found in the dataset!")
        upsample_factor = len(self.negative_samples) // len(self.positive_samples)

        # Upsample the minority class
        self.balanced_data = self.negative_samples + self.positive_samples * upsample_factor

    def __len__(self):
        return len(self.balanced_data)

    def __getitem__(self, idx):
        item = self.balanced_data[idx]
        return (
            torch.tensor(item['variables'], dtype=torch.long),
            torch.tensor(item['time_stamps'], dtype=torch.float),
            torch.tensor(item['values'], dtype=torch.float),
            torch.tensor(item['label'], dtype=torch.long),
        )

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


# Load the .npy file and preprocess the dataset
test = np.load('/Users/hanchunxu/Desktop/deep_learning/P12data/split_1/test_physionet2012_1.npy', allow_pickle=True)
processed_dataset = preprocess_dataset(test)

MAX_SEQ_LENGTH = 100  # Adjust this based on the dataset characteristics

# Split the dataset
train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
upsampled_train_data = UpsampledMortalityDataset(train_data)
# Step 1: Prepare the upsampled dataset for training
upsampled_train_loader = DataLoader(
    upsampled_train_data,  # This should be the balanced dataset
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn  # Ensure this is defined
)

# Step 2: Prepare validation loader (unmodified validation data)
val_loader = DataLoader(
    MortalityDataset(val_data),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# Step 3: Prepare test loader (unmodified test data)
test_loader = DataLoader(
    MortalityDataset(test_data),  # Use the original, unmodified test dataset
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# Step 4: Define the new model
model = NewMambaMortalityRisk(
    vocab_size=37,
    embedding_size=768,
    time_embeddings_size=32,
    num_hidden_layers=12,
    learning_rate=5e-5
)

# Step 5: Add callbacks for early stopping and checkpointing
checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

# Step 6: Initialize the Trainer
trainer = Trainer(
    max_epochs=50,  # Adjust as needed
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Step 7: Train the model
trainer.fit(model, upsampled_train_loader, val_loader)

# Step 8: Evaluate the model on the test dataset
results = trainer.test(model, dataloaders=test_loader)
print(f"Test Results: {results}")

# Step 9: Generate confusion matrix and metrics
def evaluate_model_on_test_data(model, test_loader):
    """
    Evaluate the model on the test set and generate metrics, including a confusion matrix.
    """
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            variables, time_stamps, values, labels = batch
            predictions = model(variables, time_stamps, values).squeeze()
            predicted_labels = (predictions > 0.5).cpu().numpy()  # Convert logits to binary
            all_predictions.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())

    # Generate metrics
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Data")
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=["Negative", "Positive"]))

# Run evaluation
evaluate_model_on_test_data(model, test_loader)