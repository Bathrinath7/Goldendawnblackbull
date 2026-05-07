"""
Improved Train Script for Chihuahua vs Muffin
Optimized for higher accuracy (90%+ possible with good labeling)

Run:
    python3 train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
RANDOM_SEED = 42

PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

NUM_CLASSES = 2
CLASS_NAMES = ["chihuahua", "muffin", "undefined"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("ResNet-18: Training from scratch (competition compliant)")


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[OK] Seed set to {seed}")


# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.resnet = models.resnet18(weights=None)

        features = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(

            nn.Linear(features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        features = self.resnet(x)

        return self.classifier(features)


# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        224,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def train_fn(sample):

    image = Image.open(sample["image"])

    if image.mode != "RGB":
        image = image.convert("RGB")

    return train_transform(image), sample["label"]


def val_fn(sample):

    image = Image.open(sample["image"])

    if image.mode != "RGB":
        image = image.convert("RGB")

    return val_transform(image), sample["label"]


# ============================================================================
# METRICS
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):

    labels = batch[1].to(device)

    predictions = predictor_output.forward

    softmax_output = F.softmax(predictions, dim=1)

    predicted_indices = torch.argmax(predictions, dim=1)

    confidence = torch.gather(
        softmax_output,
        1,
        predicted_indices.unsqueeze(1)
    ).squeeze(1)

    accuracy = (predicted_indices == labels).float()

    valid_labels = labels < predictions.shape[1]

    cross_entropy_loss = torch.ones_like(
        labels,
        dtype=torch.float32
    )

    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(
        reduction="none"
    )(
        predictions[valid_labels],
        labels[valid_labels]
    )

    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


# ============================================================================
# TRAINING
# ============================================================================

BEST_MODEL_FILENAME = "best_model.pth"


def train():

    set_seed(RANDOM_SEED)

    base_path = Path(__file__).parent

    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )

    print(f"[OK] Registered data path")

    # ------------------------------------------------------------------------
    # LOAD TABLES
    # ------------------------------------------------------------------------

    print("\nLoading 3LC tables...")

    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()

    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val",
    ).latest()

    print(f"Train samples: {len(train_table)}")
    print(f"Val samples: {len(val_table)}")

    class_names = list(
        train_table.get_simple_value_map("label").values()
    )

    print(f"Classes: {class_names}")

    train_table.map(train_fn).map_collect_metrics(val_fn)

    val_table.map(val_fn)

    train_sampler = train_table.create_sampler(
        exclude_zero_weights=True
    )

    train_dataloader = DataLoader(
        train_table,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_table,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    model = ResNet18Classifier(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # ------------------------------------------------------------------------
    # 3LC RUN
    # ------------------------------------------------------------------------

    run = tlc.init(
        project_name=PROJECT_NAME,
        description="Improved Chihuahua vs Muffin Training",
    )

    metric_schemas = {
        "loss": tlc.Schema(
            description="Cross entropy loss",
            value=tlc.Float32Value()
        ),

        "predicted": tlc.CategoricalLabelSchema(
            display_name="predicted label",
            classes=class_names
        ),

        "accuracy": tlc.Schema(
            description="Per-sample accuracy",
            value=tlc.Float32Value()
        ),

        "confidence": tlc.Schema(
            description="Prediction confidence",
            value=tlc.Float32Value()
        ),
    }

    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=metric_schemas,
    )

    indices_and_modules = list(
        enumerate(model.resnet.named_modules())
    )

    resnet_fc_layer_index = next(
        (
            i for i, (n, _) in indices_and_modules
            if n == "fc"
        ),
        len(indices_and_modules) - 1
    )

    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(
        layers=[resnet_fc_layer_index]
    )

    predictor = tlc.Predictor(
        model,
        layers=[resnet_fc_layer_index]
    )

    # ------------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------------

    best_val_accuracy = 0.0
    best_model_state = None

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):

        model.train()

        running_correct = 0
        running_total = 0
        running_loss = 0.0

        for images, labels in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(1)

            running_correct += (
                preds == labels
            ).sum().item()

            running_total += labels.size(0)

        train_acc = 100 * running_correct / running_total

        # --------------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_dataloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                preds = outputs.argmax(1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total

        scheduler.step()

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )

        print(
            f"Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # --------------------------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            print("NEW BEST MODEL!")

        tlc.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "val_accuracy": val_accuracy,
        })

    # ------------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"BEST VALIDATION ACCURACY: {best_val_accuracy:.2f}%")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = base_path / BEST_MODEL_FILENAME

    torch.save(model.state_dict(), model_path)

    print(f"\n[OK] Best model saved to {model_path}")

    # ------------------------------------------------------------------------
    # METRICS COLLECTION
    # ------------------------------------------------------------------------

    print("\nCollecting metrics...")

    model.eval()

    tlc.collect_metrics(
        train_table,
        predictor=predictor,
        metrics_collectors=[
            classification_metrics_collector,
            embeddings_metrics_collector
        ],
        split="train",
        dataloader_args={
            "batch_size": BATCH_SIZE,
            "num_workers": 0
        },
    )

    print("\nReducing embeddings...")

    try:

        run.reduce_embeddings_by_foreign_table_url(
            train_table.url,
            method="umap",
            n_neighbors=10,
            min_dist=0.1,
            n_components=2,
        )

        print("[OK] Embeddings reduced")

    except Exception as e:

        print(f"Embedding reduction failed: {e}")

    run.set_status_completed()

    print("\nTRAINING COMPLETE")
    print("Run: 3lc service")


if __name__ == "__main__":
    train()
