import torch
import json
import random
import numpy as np
from time import time
from argparse import ArgumentParser

from helper import Helper
from dataset import Dataset, PadCollate
from model import ContextualRescorer
from evaluate import coco_eval, write_validation_results
from logger import EarlyStopping, Logger, LrScheduler
from visualize import visualize_model
from mmcv import ProgressBar


def training_step(model, optimizer, input_batch, tgt_batch, lengths):
    model.train()
    optimizer.zero_grad()
    mask = (tgt_batch != -1).float()
    pred = model.forward(input_batch, lengths, mask)
    loss = model.loss(pred, tgt_batch, input_batch)
    loss.backward()
    optimizer.step()

    # Count statistics
    corrects = (pred.round() == tgt_batch.round()).sum()
    total = (tgt_batch != -1).sum()
    return float(loss), int(corrects), int(total)


def validate(dataloader, model):
    loss, corrects, total_predictions = 0, 0, 0
    model.eval()

    for i, (input_tensor, target_tensor, lengths) in enumerate(dataloader):
        mask = (target_tensor != -1).float()
        predictions = model.forward(input_tensor, lengths, mask)
        loss += model.loss(predictions, target_tensor, input_tensor).item()
        corrects += (predictions.round() == target_tensor.round()).sum().item()
        total_predictions += (target_tensor != -1).sum().item()
    accuracy = corrects / total_predictions * 100
    return loss / (i + 1), accuracy


def main(config, params, dataset):

    helper = Helper("/mnt/annotations/instances_val2017.json")
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start = time()
    print("Loading train dataset...")
    train_dataset = Dataset("/mnt/preprocessed/preprocessed_train2017_" + dataset + ".pt")
    torch.cuda.empty_cache()

    print("Loading validation set...")
    val_dataset = Dataset("/mnt/preprocessed/preprocessed_val2017_" + dataset + ".pt")
    torch.cuda.empty_cache()
    print("Loaded validation set. (t=%.1f seconds)" % (time() - start))

    val_params = {"batch_size": params["val_batch_size"], "collate_fn": PadCollate()}
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_params)

    train_params = {
        "batch_size": params["batch_size"],
        "shuffle": True,
        "collate_fn": PadCollate(shuffle_rate=params["shuffle_rate"]),
    }
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_params)

    # Train loop
    model = ContextualRescorer(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = LrScheduler(optimizer)
    logger = Logger(config, params, dataset=dataset)
    early_stopping_params = {"mode": "max", "patience": 20, "delta": 0.0001}
    early_stopper = EarlyStopping(**early_stopping_params)

    start = time()
    for epoch in range(params["n_epochs"]):
        loss, corrects, total = 0, 0, 0
        prog_bar = ProgressBar(len(train_dataloader))
        for i, (input_batch, target_batch, lengths) in enumerate(train_dataloader):
            batch_loss, corrects_, total_ = training_step(
                model, optimizer, input_batch, target_batch, lengths
            )
            loss += batch_loss
            corrects += corrects_
            total += total_
            prog_bar.update()

        loss = loss / (i + 1)
        accuracy = corrects / total * 100

        # Measure loss and accuracy on validation set
        val_loss, val_accuracy = validate(val_dataloader, model)

        # Evaluate the AP on the validation set
        model.eval()
        print("\n --> Evaluating AP")
        write_validation_results(val_dataset, model, helper)
        stats = coco_eval()
        ap = stats[0]
        print("AP: {} \n\n".format(ap))

        if scheduler.step(ap):
            print(" --> Backtracking to best model")
            model.load_state_dict(logger.best_model)

        # Logging and early stopping
        logger.epoch(model, loss, accuracy, val_loss, val_accuracy, ap, optimizer.param_groups[0]["lr"])
        if early_stopper.step(ap):
            print("	--> Early stopping")
            break

    logger.close()
    visualize_model(helper, params, logger.best_model, val_dataset)
    print(config)


if __name__ == "__main__":    
    # Set random seeds
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    parser = ArgumentParser()
    parser.add_argument(
        "config",
        default="cfg_000.json",
        help="File with networks parameters in .cfg format",
    )
    parser.add_argument(
        "dataset", default="cascade101", help="Train/val dataset name"
    )

    args = parser.parse_args()

    with open(args.config) as json_file:
        params = json.load(json_file)

    main(args.config, params, args.dataset)
