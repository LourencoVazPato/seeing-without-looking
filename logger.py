import copy
import time
import os
import json
import torch
# from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, cfg_file, params, print_=True, dataset='cascade101'):
        
        self.folder = "logs/" + cfg_file.split('.')[0].split('/')[-1]
        self.params = params
        datetime = time.strftime("%Y%m%d%H%M", time.gmtime())

        print("\n               NEW MODEL\n", self.folder)

        # self.writer = SummaryWriter(log_dir=self.folder)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.aps = []
        self.learning_rates = []
        self.n_epoch = 0
        self.best = None
        self.best_epoch = None
        self.best_model = None
        self.start = time.time()
        self.dataset = dataset
        self._init_is_better()

    def epoch(self, model, loss, acc, val_loss, val_acc, ap, lr):
        is_best = False
        if self.best is None:
            self.best = ap
            self.best_epoch = self.n_epoch
            self.best_model = copy.deepcopy(model.state_dict())
            is_best = True
        
        if self.is_better(ap, self.best):
            self.best = ap
            self.best_epoch = self.n_epoch
            self.best_model = copy.deepcopy(model.state_dict())
            is_best = True

        self._print_progress(loss, acc, val_loss, val_acc, ap, is_best=is_best)
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.aps.append(ap)
        self.learning_rates.append(lr)
        self.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # self.writer.add_scalar("loss", loss, global_step=self.n_epoch)
        # self.writer.add_scalar("accuracy", acc, global_step=self.n_epoch)
        # self.writer.add_scalar("AP", ap, global_step=self.n_epoch)
        self.n_epoch += 1

    def close(self):
        # self.writer.close()
        print(" -> Best:", round(self.best, 4), ". Best epoch:", self.best_epoch)
        stats = {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "dataset": self.dataset,
            "ap": self.aps,
            "train_loss": self.train_losses,
            "train_accuracy": self.train_accuracies,
            "val_loss": self.val_losses,
            "val_accuracy": self.val_accuracies,
            "num_parameters": self.num_parameters,
            "learning_rate": self.learning_rates,
            "epochs": self.n_epoch,
            "time": time.time() - self.start,
        }

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        with open(self.folder + "/training.json", "w") as json_file:
            json.dump(stats, json_file)
        with open(self.folder + "/params.json", "w") as json_file:
            json.dump(self.params, json_file)
        torch.save(self.best_model, self.folder + "/model.pt")

    def printBold(self, text):
        print("\033[1m" + str(text) + "\033[0m")

    def _print_progress(self, loss, acc, val_loss, val_acc, ap, is_best=False):
        if is_best:
            print_fn = self.printBold
        else:
            print_fn = print

        print_fn(
            "---------- epoch %d completed ---------- t=%.1fm"
            % (self.n_epoch, (time.time() - self.start) / 60)
        )
        print_fn("Training set: Loss: %.4f  Acc: %.2f %%" % (loss, acc))
        print_fn("Validation  : Loss: %.4f  Acc: %.2f %%" % (val_loss, val_acc))
        print_fn("		AP : %.2f" % (ap * 100))

    def _init_is_better(self, mode="ap"):
        if mode not in {"loss", "acc", "ap"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "loss":
            self.is_better = lambda a, best: a < best
        else:
            self.is_better = lambda a, best: a > best


class EarlyStopping(object):
    def __init__(self, mode="max", patience=5, delta=0.0001):
        self.best = None
        self.counter = 0
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.is_better = None
        self._init_is_better(mode)

    def _init_is_better(self, mode):
        if mode == "min":
            self.is_better = lambda a, best: a <= best - self.delta
        else:
            self.is_better = lambda a, best: a >= best + self.delta

    def step(self, ap):
        if self.best is None:
            self.best = ap
            return False

        if self.is_better(ap, self.best):
            self.counter = 0
            self.best = ap
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False


class LrScheduler(object):
    def __init__(self, optimizer, return_to_max=True):
        self.params = {
            "mode": "max",
            "factor": 0.2,
            "patience": 4,
            "cooldown": 1,
            "threshold": 0.0001,
            "threshold_mode": "abs",
            "verbose": True,
        }
        self.return_to_max = return_to_max
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.params
        )
        self.lr = self.get_lr()

    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]["lr"]

    def step(self, metrics):
        self.scheduler.step(metrics)

        if self.get_lr() < self.lr:
            self.lr = self.get_lr()
            return True
        return False
