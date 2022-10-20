import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, out_features=5):
        super().__init__()
        # First Convolution Block with Relu and Batch Norm.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)


        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                 padding=(1, 1))  # determine this later! padding=1 because
        # kernel size probably doesn't fit the input evenly

        # Define proportion of nodes to dropout
        self.dropout = nn.Dropout(p=0.1)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=out_features)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)

        x = x.view(x.shape[0], -1)  # shape [1, num_features]
        # Linear layer
        x = self.lin(x)  #

        # Final output (don't softmax, taken care of in nn.CrossEntropyLoss()
        return x  # length 5, value per class



def weights_init(m):
    for name, param in m.named_parameters():
        if 'bn' not in name:
            if 'bias' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
        # default for lstm is He (see https://stackoverflow.com/questions/65606553/how-pytorch-model-layer-weights-get-initialized-implicitly )



class LRCN(torch.nn.Module):
    def __init__(self, lstm_input_size=64, hidden_size=128, dropout_rate=0.4, num_classes=5):
        super(LRCN, self).__init__()
        self.lstm_input_size = lstm_input_size
        self.cnn = AudioClassifier(out_features=lstm_input_size)  # need to figure out num out_features
        # batch first: data formatted in (batch, seq, feature)
        self.lstm = torch.nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

        # Define proportion of nodes to dropout
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        # softmax layer
        # self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, batch):
        data, lengths = batch
        # data dimension: [B, T, C, H, W]
        # chunk_seq dimension: [B, T, CNN's output dimension]
        # it is used to store all audio features
        chunk_seq = torch.empty(size=(data.size()[0], data.size()[1], self.lstm_input_size), device='cuda')

        for t in range(0, data.size()[1]):
            chunk = data[:, t, :, :, :]  # e.g. [16, 20, 1, 64, 38] = the 20th chunk in the sequence
            cnn_out = self.cnn(chunk)
            chunk_seq[:, t, :] = cnn_out
        # print(f'chunk_seq shape = {chunk_seq.shape}')

        # PackedSequence is NamedTuple with 2 attributes: data and batch_sizes.
        # Pack the padded data so LSTM does not see padding
        chunk_seq_packed = pack_padded_sequence(chunk_seq, lengths.cpu(), batch_first=True)
        # chunk_seq_packed.data dimensions: (batch_sum_seq_len, CNN's output dimension)

        packed_out, (_, _) = self.lstm(chunk_seq_packed)
        # packed_out.data dimensions: (batch_sum_seq_len, hidden_size)

        # change the packed_out back into a padded tensor for the linear layer
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        # unpacked dimension: (batch_size, seq_length, hidden_size)

        x = self.dropout(unpacked)

        x = self.linear(x)
        # x dimension: (batch_size, seq_length, classNum)
        # print(f'linear out shape = {x.shape}')

        # with softmax - requires nn.NLLLoss function:
        # x = self.logsoftmax(x)

        # get frame-wise's mean but ignore padding based on original lengths
        out = torch.stack([torch.mean(x[i, 0:leng], dim=0) for i, leng in enumerate(lengths)])
        # out dimension：(batch, class_Num)
        return out

# packing - https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
# https://github.com/BizhuWu/LRCN_PyTorch/blob/main/model.py
# https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
# https://github.com/BizhuWu/LRCN_PyTorch/blob/main/model.py
# from https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/