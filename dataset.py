import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random


class Dataset(Dataset):
    """ Loads a preprocessed dataset of input-target pairs, inherits from torch Dataset class"""

    def __init__(self, path):
        self.inputs = {}
        self.targets = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets = torch.load(path)
        self.ids = list(inputs.keys())

        for id_ in self.ids:
            self.inputs[id_] = inputs[id_]  # .cpu()
            self.targets[id_] = targets[id_].view(-1, 1)  # .cpu()
        print("Loaded", path)

    def load(self, path):
        inputs, targets = torch.load(path)
        new_ids = list(inputs.keys())
        self.ids += new_ids
        for id_ in new_ids:
            self.inputs[id_] = inputs[id_]  # .cpu()
            self.targets[id_] = targets[id_].view(-1, 1)  # .cpu()
        print("Loaded", path)

    def get_id(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.get_id(index)
        input_ = self.inputs[id_].to(self.device)
        target = self.targets[id_].to(self.device)
        return input_, target


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, shuffle_rate=0):
        self.shuffle_rate = shuffle_rate

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label) tuples
        return:
            xs - a tensor of all examples in the batch after padding
            ys - a tensor of all labels in batch
        """
        input_tensors, target_tensors, lengths = [], [], []

        if random.random() < self.shuffle_rate:
            self.shuffle_batch(batch)  # shuffle the bounding boxes during training

        for i, (inpt, targt) in enumerate(batch):
            input_tensors.append(inpt)
            target_tensors.append(targt)
            lengths.append(inpt.size(0))

        lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
        padded_input = pad_sequence(input_tensors, batch_first=True)
        padded_target = pad_sequence(target_tensors, batch_first=True, padding_value=-1)  # padding tag is target = -1
        return padded_input, padded_target, lengths

    def shuffle_batch(self, batch):
        """
        Permute the bboxes around, batch is a list of (input_tensor, target_tensor) tuples
        """
        for i, (inpt, targt) in enumerate(batch):
            seq_len = inpt.size(0)
            order = np.random.permutation(seq_len)
            batch[i] = (inpt[order, :], targt[order])

    def __call__(self, batch):
        return self.pad_collate(batch)
