import os
import sys
import torch

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

#from apex.apex.apex.amp import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

TRAINING_CSV = "Chapter-9/input/siim_png/train.csv"

TRAINING_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

EPOCHS = 10

ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda:0"

def train(dataset, data_loader, model, criterion, optimizer):
    model.train()
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    for d in tk0:
        inputs = d["image"]
        targets = d["mask"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        """with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()"""
        optimizer.step()
    
    tk0.close()

def evaluate(dataset, data_loader, model, criterion):
    model.eval()
    final_loss = 0
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]

            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            final_loss += loss

    tk0.close()
    return final_loss / num_batches

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_CSV)
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )

    training_images = df_train.ImageId.values
    validation_images = df_valid.ImageId.values

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes = 1,
        activation=None,
    )

    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    model.to(DEVICE)

    train_dataset = SIIMDataset(
        image_ids=training_images,
        transform=True,
        preprocessing_fn=prep_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    valid_dataset = SIIMDataset(
        image_ids=validation_images,
        transform=True,
        preprocessing_fn=prep_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )
    """model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", verbosity=0
    )"""
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, criterion, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model, criterion)
        scheduler.step(val_score)
        print(f"Epoch: {epoch}, val_score: {val_score}")
        print("\n")