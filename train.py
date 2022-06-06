import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from util import MyCollate
from model import BaseModel
from vocab import Vocabulary


class TextDataset(Dataset):
    def __init__(self, data_dir, mode, vocab_size):
        self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        self.x = self.df['text'].values
        self.y = self.df['label'].values

        self.sentences = None
        self.labels = None

        x_train, x_valid, y_train, y_valid = train_test_split(self.x, self.y, test_size=0.1, random_state=44)

        if mode == 'train':
            self.sentences = x_train
            self.labels = y_train
        elif mode == 'valid':
            self.sentences = x_valid
            self.labels = y_valid

        # Initialize dataset Vocabulary object and build our vocabulary
        self.sentences_vocab = Vocabulary(vocab_size)
        self.labels_vocab = Vocabulary(vocab_size)
    
        self.sentences_vocab.build_vocabulary(self.sentences, mode=mode)
        self.labels_vocab.build_vocabulary(self.labels, add_unk=False, mode=mode)

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
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle,
                        collate_fn = MyCollate(pad_idx=pad_idx, batch_first=batch_first))   # MyCollate class runs __call__ method by default
    return loader


def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred==label).item()


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


def train(args, data_loader, valid_loader, model):
    global_train_loss = []
    global_train_acc = []
    global_valid_acc = []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = np.Inf
    
    for epoch in range(args.num_epochs):
        train_losses = [] 
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        model.train()
        for i, (text, label) in enumerate(tqdm(data_loader)):
            input_lengths = torch.LongTensor([torch.max(text[j,:].nonzero())+1 for j in range(text.size(0))])
            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            text = text[sorted_idx].to(args.device)
            label = label.squeeze()
            label = label[sorted_idx].to(args.device)    
            
            optimizer.zero_grad()

            output, _ = model(text, input_lengths)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total

        global_train_loss.append(epoch_train_loss)
        global_train_acc.append(epoch_train_acc)
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}%'.format(epoch_train_acc*100))

        pred, true = test(args, valid_loader, model)
        valid_accuracy = (true == pred).sum() / len(pred)
        global_valid_acc.append(valid_accuracy)
        print("Valid Accuracy : {:.3f}%\n".format(valid_accuracy*100))

        # Save Model
        torch.save(model.state_dict(), f'model{epoch+1}.pt')
        print('Train loss ({:.6f} --> {:.6f}).  Saving model ...\n'.format(min_loss, epoch_train_loss))
        min_loss = epoch_train_loss

        torch.save(model.embedding, f'embedding{epoch+1}.pt')
        print('Saving embedding tensor\n\n')
    print(f'train_loss list = {global_train_loss}')
    print(f'train_acc list = {global_train_acc}')
    print(f'valid_acc list = {global_valid_acc}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', type=bool, default=True, help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of epochs to train for (default: 5)")
    
    args = parser.parse_args()

    # Model hyperparameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 256 # embedding dimension
    hidden_dim = 32  # hidden size of RNN
    num_layers = 2

    # Make Train & valid Loader
    train_dataset = TextDataset(args.data_dir, 'train', args.vocab_size)
    valid_dataset = TextDataset(args.data_dir, 'valid', args.vocab_size)

    args.pad_idx = train_dataset.sentences_vocab.wtoi['<PAD>']

    train_loader = make_data_loader(train_dataset, args.batch_size, args.batch_first, shuffle=True)
    valid_loader = make_data_loader(valid_dataset, args.batch_size, args.batch_first, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = model.to(device)

    # Training The Model
    train(args, train_loader, valid_loader, model)