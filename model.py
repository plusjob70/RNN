from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first: bool):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_dim = hidden_dim

        print('create new embedding matrix')

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=batch_first)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_size)

    def load_embedding(self, embedding_file, device):
        print('load embedding matrix')
        self.embedding = nn.Embedding.from_pretrained(torch.load(embedding_file, map_location=device).weight, freeze=True, padding_idx=0)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x, input_lengths):
        embedding = self.embedding(x)

        packed_input = pack_padded_sequence(embedding, input_lengths.tolist(), batch_first=self.batch_first)
        packed_output, hidden = self.rnn(packed_input)
        
        h_n = hidden[0]
        self.dropout(h_n)
        output = self.fc(h_n)
        
        return output, hidden