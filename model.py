import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextualRescorer(nn.Module):
    def __init__(self, params):
        super(ContextualRescorer, self).__init__()

        self.hidden_size = params['hidden_size']
        self.input_size = params['input_size']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_layers = params['num_layers']
        self.directions = 2 if params['bidirectional'] else 1
        self.skip_connection = params['skip_connection']
        self.embedding_layer = params['embedding']
        self.attention_type = params['attention_type']
        self.loss_type = 'mse' 

        # Embedding layer
        if self.embedding_layer:
            self.embedding_size = params['embedding_size']	
            self.input_size = self.input_size - 80 + self.embedding_size
            self.embedding = nn.Embedding(80, self.embedding_size)

        # Initialize hidden vectors
        self.h0 = nn.Parameter(
            torch.zeros(self.num_layers * self.directions, 1, self.hidden_size),
            requires_grad=True,
        )

        # Network layers
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=params['bidirectional'],
            batch_first=True,
            dropout=params['dropout'],
        )

        if self.attention_type != "none":
            layer_size = self.hidden_size * self.directions * 2
        else:
            layer_size = self.hidden_size * self.directions

        if self.skip_connection:
            layer_size = layer_size + self.input_size

        self.linear1 = nn.Linear(layer_size, 80)
        self.linear2 = nn.Linear(80, 1, bias=False)
        self.relu = nn.ReLU()
        
        # Attention layers
        if self.attention_type == "general":
            self.Wa = nn.Linear(self.hidden_size * self.directions, self.hidden_size * self.directions, bias=True)
        if self.attention_type == "additive":
            self.Wa = nn.Linear(self.hidden_size * self.directions * 2, 256, bias=True)
            self.va = nn.Linear(256, 1, bias=False)

    def init_hidden(self, batch_size=1):
        h0 = self.h0.repeat(1, batch_size, 1)
        return h0

    def forward(self, input_, lengths, mask):

        batch_size, _, _ = input_.size()

        # Embedding layer
        if self.embedding_layer:
            cat = input_[:, :, 1:81].argmax(dim=2)
            embeddings = self.embedding(cat)
            scores = input_[:, :, :1]
            bbox = input_[:, :, -4:]
            input_ = torch.cat((scores, embeddings, bbox), dim=2)
        
        # Pack the sequence and propagate through RNN
        h0 = self.init_hidden(batch_size=batch_size)
        packed_input = pack_padded_sequence(
            input_, lengths, batch_first=True, enforce_sorted=False
        )
        hidden, _ = self.rnn(packed_input, h0)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)  # unpack sequence

        # Classifier layer
        if self.attention_type != "none":
            context = self.attention(hidden, mask)
            output = self.classifier(hidden, context, input_)
        else:
            output = self.classifier(hidden, "none", input_)
        return output

    def classifier(self, hidden, context, input_):
        if self.skip_connection:
            if self.attention_type == "none":
                hidden = torch.cat((hidden, input_), dim=2)
            else:
                hidden = torch.cat((hidden, context, input_), dim=2)
        else:
            if self.attention_type != "none":
                hidden = torch.cat((hidden, context), dim=2)
        
        hidden = self.relu(self.linear1(hidden))
        hidden = self.linear2(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden
    
    def attention(self, hidden, mask):
        B, L, H = hidden.size()
        if self.attention_type == 'dot_product':
            scores = self.dot_product_attn(hidden)
        elif self.attention_type == 'general':
            scores = self.general_attn(hidden)
        elif self.attention_type == 'additive':
            scores = self.additive_attn(hidden)
        else:
            scores = self.scaled_dot_product_attn(hidden, mask)
        msk = (mask == 0).view(B, 1, L).float()
        scores = scores - msk * 100000  # mask out padding bboxes
        alpha = torch.softmax(scores, dim=2)
        context = torch.bmm(alpha, hidden)  # [B, L, h]
        return context

    def additive_attn(self, hidden):
        B, L, H = hidden.size()
        hidden = torch.cat((hidden.unsqueeze(2).expand(-1, -1, L, -1), hidden.unsqueeze(1).expand(-1, L, -1, -1)), dim=3)  # [B, L, L, 2H]
        hidden = self.Wa(hidden)  # [B, L, L, m]
        hidden = torch.tanh(hidden)
        hidden = self.va(hidden).squeeze(3)  # [B, L, L]
        return hidden

    def general_attn(self, hidden):
        hs = self.Wa(hidden)  # [B, L, H] * [., H, H]
        scores = torch.bmm(hidden, hs.transpose(1, 2))  # [B, L, H] * [B, H, L]
        return scores  # [B, L, L]

    def dot_product_attn(self, hidden):
        scores = torch.bmm(hidden, hidden.transpose(1, 2))  # [B, L, H] * [B, H, L]
        return scores  # [B, L, L]

    def scaled_dot_product_attn(self, hidden, mask):
        scores = torch.bmm(hidden, hidden.transpose(1, 2))  # [B, L, L]
        ln = mask.sum(dim=1).sqrt().unsqueeze(2)  # [B, 1, 1]
        return scores / ln  # [B, L, L]

    def loss(self, predictions, targets, input_):
        assert (
            predictions.shape == targets.shape
        ), "Predictions and targets have different shape"
        # create a mask to compute the loss (-1 is the target padding value)
        mask = (targets != -1).float()

        if self.loss_type == 'mse':
            return torch.sum(mask * (predictions - targets) ** 2) / mask.sum()
        
        elif self.loss_type == 'bce':
            # Weighted BCE Loss/log loss
            # confidence = input_[:,:,0].unsqueeze(2)
            # weight = torch.exp(2*targets)
            # 'sum' all elements and divide by the elements in mask
            criterion = nn.BCELoss(weight=mask, reduction='sum')
            return criterion(predictions, targets) / mask.sum()
