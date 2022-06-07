import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from util import MyCollate
from model import BaseModel
from vocab import Vocabulary
# from sklearn.metrics import classification_report


class TextDataset(Dataset):

    def __init__(self, data_dir, mode, vocab_size):
        self.df = pd.read_csv(os.path.join(data_dir, mode + '.csv'))

        self.sentences = self.df['text'].values
        self.labels = self.df['label'].values

        # Initialize dataset Vocabulary object and build our vocabulary
        self.sentences_vocab = Vocabulary(vocab_size)
        self.labels_vocab = Vocabulary(vocab_size)

        self.sentences_vocab.build_vocabulary(self.sentences, mode='test')
        self.labels_vocab.build_vocabulary(self.labels, add_unk=False, mode='test')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # numericalize the sentence ex) ['cat', 'in', 'a', 'bag'] -> [2,3,9,24,22]
        numeric_sentence = self.sentences_vocab.sentence_to_numeric(sentence)
        numeric_label = self.labels_vocab.sentence_to_numeric(label)

        return torch.tensor(numeric_sentence), torch.tensor(numeric_label)


def make_data_loader(dataset, batch_size, batch_first, shuffle=True):   # increase num_workers according to CPU
    # get pad_idx for collate fn
    pad_idx = dataset.sentences_vocab.wtoi['<PAD>']
    # define loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=MyCollate(pad_idx=pad_idx, batch_first=batch_first))     # MyCollate class runs __call__ method by default
    return loader


def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    model.eval()

    with torch.no_grad():
        for i, (text, label) in enumerate(tqdm(data_loader)):
            input_lengths = torch.LongTensor([torch.max(text[j,:].nonzero())+1 for j in range(text.size(0))])
            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)

            text = text[sorted_idx].to(args.device)

            label = label.squeeze()
            label = label[sorted_idx].to(args.device)            

            output, _ = model(text, input_lengths)
            
            output = output.argmax(dim=-1)
            output = output.detach().cpu().numpy()
            pred = np.append(pred,output, axis=0)
            
            label = label.detach().cpu().numpy()
            true = np.append(true, label, axis=0)

    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=32000, help="maximum vocab size")
    parser.add_argument('--batch_first', type=bool, default=True, help="If true, then the model returns the batch first")
    parser.add_argument('--test_data',  type=str, default='test')
    ##
    parser.add_argument('--epoch',  type=int, default=12)
    ##

    args = parser.parse_args()

    """
    TODO: You MUST write the same model parameters as in the train.py file !!
    """
    # Model parameters
    input_size = args.vocab_size
    output_size = 4      # num of classes
    embedding_dim = 512  # embedding dimension
    hidden_dim = 32      # hidden size of RNN
    num_layers = 1

    # Make Test Loader
    test_dataset = TextDataset(args.data_dir, args.test_data, args.vocab_size)
    args.pad_idx = test_dataset.sentences_vocab.wtoi['<PAD>']
    test_loader = make_data_loader(test_dataset, args.batch_size, args.batch_first, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=True)

    model.load_state_dict(torch.load('model.pt', map_location=device))
    #model.load_embedding('embedding.pt', device=device)

    model = model.to(device)
    
    print(test_dataset.labels_vocab.itow)
    target_names = [w for i, w in test_dataset.labels_vocab.itow.items()]
    # Test The Model
    pred, true = test(args, test_loader, model)
    accuracy = (true == pred).sum() / len(pred)

    print("Test Accuracy : {:.5f}".format(accuracy))
    print('\n\n')


    # Save result
    strFormat = '%10s%10s\n'

    with open('result.txt', 'w') as f:
        f.write('Test Accuracy : {:.5f}\n'.format(accuracy))
        f.write('true label  |  predict label \n')
        f.write('-------------------------- \n')
        for i in range(len(pred)):
            f.write(strFormat % (test_dataset.labels_vocab.itow[true[i]], test_dataset.labels_vocab.itow[pred[i]]))

    # print(classification_report(true, pred, target_names=target_names))
    