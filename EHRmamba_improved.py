import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, labels):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(predictions.squeeze(), labels.float())
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class ImprovedMambaMortalityRisk(pl.LightningModule):
    def __init__(
        self,
        vocab_size=37,
        embedding_size=768,
        time_embeddings_size=32,
        num_hidden_layers=12,
        learning_rate=5e-5,
        positive_class_weight=5.0,  # Adjusted Positive Class Weight
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.positive_class_weight = positive_class_weight
        self.loss_function = FocalLoss(alpha=self.positive_class_weight)

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
            nn.Dropout(0.3),
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, variables, time_stamps, values):
        variable_embeds = self.variable_embeddings(variables)
        variable_embeds = variable_embeds.unsqueeze(1).repeat(1, time_stamps.shape[1], 1, 1).mean(dim=2)

        time_embeds = self.time_embeddings(time_stamps.long())
        time_embeds = self.time_project(time_embeds)

        value_embeds = self.value_layer(values.unsqueeze(-1))
        value_embeds = value_embeds.mean(dim=2)

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

        predicted_labels = (predictions.squeeze() > 0.7).float()
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

        predicted_labels = (predictions.squeeze() > 0.7).float()
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

# Collate function to handle varying sequence lengths
def collate_fn(batch):
    variables, time_stamps, values, labels = zip(*batch)

    # Pad sequences to the same length using clone().detach()
    variables = pad_sequence(
        [v.clone().detach() for v in variables], batch_first=True, padding_value=0
    )
    time_stamps = pad_sequence(
        [t.clone().detach() for t in time_stamps], batch_first=True, padding_value=0
    )
    values = pad_sequence(
        [val.clone().detach() for val in values], batch_first=True, padding_value=0
    )

    labels = torch.tensor(labels).float()
    return variables, time_stamps, values, labels



test = np.load('/Users/hanchunxu/Desktop/deep_learning/P12data/split_1/test_physionet2012_1.npy', allow_pickle=True)
processed_dataset = preprocess_dataset(test)

# Assuming `dataset` is your PyTorch Dataset
dataloader = DataLoader(
    processed_dataset,
    batch_size=32,
    collate_fn=collate_fn,
    num_workers=7,  # Increase for better performance
    pin_memory=True,  # Optional for speedup on CUDA
)


# Assuming `processed_dataset` is a list of tuples (variables, time_stamps, values, labels)

# Split the dataset
train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create DataLoaders with the custom collate function
train_loader = DataLoader(
    MortalityDataset(train_data),  # Your dataset class
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn  # Handles padding if sequences vary in length
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

# Define Model
# Trainer with Callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

model = ImprovedMambaMortalityRisk(
    vocab_size=37,
    embedding_size=768,
    time_embeddings_size=32,
    num_hidden_layers=12,
    learning_rate=5e-5,
    positive_class_weight=10.0  # Adjust weight for positive class
)

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stopping]
)

# Train and Test
trainer.fit(model, train_loader, val_loader)
test_results = trainer.test(model, dataloaders=test_loader)

# Display Results
print("Test Results:")
print(f"Loss: {test_results[0]['test_loss']:.4f}")
print(f"Accuracy: {test_results[0]['test_accuracy']:.4f}")
print(f"Precision: {test_results[0]['test_precision']:.4f}")
print(f"Recall: {test_results[0]['test_recall']:.4f}")
print(f"F1 Score: {test_results[0]['test_f1']:.4f}")
