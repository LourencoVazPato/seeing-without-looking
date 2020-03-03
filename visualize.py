import matplotlib.pyplot as plt
import json
from termcolor import colored
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from model import ContextualRescorer


def printBold(text):
    print("\033[1m" + str(text) + "\033[0m")


def visualize_model(helper, params, state_dict, dataset, n_samples=10):

    # Load category data
    with open("data/instances_val2017.json") as json_file:
        data = json.load(json_file)
    categories = data["categories"]
    categories = {cat["id"]: cat for cat in categories}
    index = list(categories.keys())
    images = data["images"]
    images = {img["id"]: img for img in images}
    del data

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ContextualRescorer(params).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1)
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        mask = (target_tensor != -1).float()
        target_tensor = target_tensor.view(-1)
        predictions = model.forward(input_tensor, [input_tensor.size(1)], mask)

        img_id = dataset.ids[i]
        print("image ID:", img_id)
        W, H = images[img_id]["width"], images[img_id]["height"]
        target_order = target_tensor.argsort(descending=True).tolist()
        rescored_order = predictions.view(-1).argsort(descending=True).tolist()
        predicted_order = input_tensor[0, :, 0].argsort(descending=True).tolist()

        print(
            "Confidence | Rescored | Target | Pred | Resc | Targ |         bbox         |  Class"
        )
        print(
            "------------------------------------------------------------------------------"
        )
        for j in range(input_tensor.size(1)):
            if predictions.round()[:, j, :]:
                attrs = ["bold"]
            else:
                attrs = []
            if predictions.round()[:, j] == target_tensor.round()[j]:
                color = "green"
            else:
                color = "red"

            pred_score = round(predictions[:, j].item(), 3)
            det_score = round(input_tensor[0, j, 0].item(), 3)
            target_iou = round(target_tensor[j].item(), 3)
            _, det_class = input_tensor[0, j, 1:81].max(0)
            x, y, w, h = input_tensor[0, j, -4:]
            det_class = categories[index[det_class]]["name"]
            string = (
                "   %.3f   |  %.3f   |  %.2f  |  %d  |  %d  |  %d  | %.1f %.1f %.1f %.1f |  %s"
                % (
                    det_score,
                    pred_score,
                    target_iou,
                    predicted_order.index(j),
                    rescored_order.index(j),
                    target_order.index(j),
                    x * W,
                    y * H,
                    w * W,
                    h * H,
                    det_class,
                )
            )
            print(colored(string, color, attrs=attrs))
        print()
        if i == n_samples:
            break


def plot_training(train_stats):
    train_losses = train_stats["train_losses"]
    train_accuracy = train_stats["train_accuracy"]
    validation_losses = train_stats["validation_losses"]
    validation_accuracy = train_stats["validation_accuracy"]
    # APs = stats['train']['APs']

    max_index = 0
    min_index = 0
    max_ = 0
    min_ = 100
    for i, acc in enumerate(validation_accuracy):
        if acc >= max_:
            max_ = acc
            max_index = i + 1
    for i, loss in enumerate(validation_losses):
        if loss <= min_:
            min_ = loss
            min_index = i + 1

    print("Maximum validation accuracy:", round(max_, 4), "% at epoch", max_index)
    print("Minimum validation loss:", round(min_, 4), "at epoch", min_index)

    plt.figure(figsize=(15, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.legend(["training loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy)
    plt.plot(validation_accuracy)
    plt.legend(["training accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy [ %]")
    # AP
    # plt.figure()
    # plt.plot(APs)
    # plt.legend(['validation AP'])
    # plt.xlabel('epoch (x4)')
    # plt.ylabel('AP')
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model training and results."
    )
    parser.add_argument(
        "folder",
        help="folder containing stats.json with model parameters and model.pt containing the trained model",
    )
    parser.add_argument(
        "--ablation",
        default=["None"],
        nargs="*",
        help="Ablation to apply to data (default: None)",
    )
    args = parser.parse_args()

    with open(args.folder + "stats.json") as file_:
        stats = json.load(file_)

    # Plot training and validation curves (loss and accuracy)
    plot_training(stats["train"])

    state_dict = torch.load(args.folder + "model.pt")
    # Load validation dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_dataset = Dataset("data/preprocessed_val2017_ious_cascade101.pt", device)
    stats["hyperparams"]["input_size"] = 85
    visualize_model(
        state_dict, validation_dataset, stats["hyperparams"], ablation=args.ablation
    )


if __name__ == "__main__":
    main()
