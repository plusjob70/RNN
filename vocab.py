from collections import Counter
from itertools import compress
import pandas as pd
import re


class Vocabulary:
    def __init__(self, vocab_size):
        """
        TODO: Change vocabulary code as you need. (e.g. tokenizer, using stopword, etc.)
        
        vocab_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        """

        # <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        # <UNK> -> words which are not found in the vocab are replace by this token
        self.vocab_size = vocab_size
        self.itow = {0: '<PAD>', 1: '<UNK>'}
        self.wtoi = {w: i for i, w in self.itow.items()}
        self.stop_words = None
        
        try:
            with open('./stopwords.txt', mode='r') as f:
                self.stop_words = [line.rstrip() for line in f.readlines()]
        except FileNotFoundError:
            print('stopwords.txt 없음', flush=True)
            self.stop_words = []

    def __len__(self):
        return len(self.itow)

    @staticmethod
    def __isfloat(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # word : 1-word
    def regularize(self, word):
        word = re.sub(r"[\']s|’s|[(|)|,|\"|\'|‘|’|`|“|”|:|;|\[|\]|?|!]|(\.(?=[\s\n\r”\"]|$))", '', word).lower().strip()
        if word.isdigit():
            word = '<DIG>'
        elif self.__isfloat(word):
            word = '<FLO>'
        elif '°c' in word:
            word = '<TEM>'
        elif '°' in word:
            word = '<DEG>'
        elif '%' in word:
            word = '<PER>'
        return word

    # text : 1-sentence
    def tokenizer(self, text):
        return [self.regularize(tok) for tok in text.split(' ')]

    def load_vocabulary(self, filename):
        df = pd.read_csv(f'./Data/{filename}')
        self.itow = dict(zip(df['idx'], df['word']))
        self.wtoi = dict(zip(df['word'], df['idx']))

    def save_vocabulary(self, filename):
        idx = self.itow.keys()
        word = self.itow.values()

        df = pd.DataFrame(data={'idx': idx, 'word': word})
        df.to_csv(f'./Data/{filename}')

    # type(sentences) : numpy.ndarray
    def build_vocabulary(self, sentences, add_unk=True, mode='train'):
        if mode == 'train':
            idx = 2  # index from which we want our dict to start. We already used 2 indexes for pad and unk

            if add_unk == False:  # build vocab for label
                self.wtoi = {}
                idx = 0

            word_list = [word
                for sentence in sentences
                    for word in self.tokenizer(sentence)
                        if word
            ]

            freq = Counter(word_list)
            freq_ = sorted(freq, key=freq.get, reverse=True)

            # remove stop words
            freq_ = list(compress(freq_, [0 if word in self.stop_words else 1 for word in freq_]))
            
            for word in freq_[:self.vocab_size - idx]:
                self.wtoi[word] = idx
                self.itow[idx] = word
                idx += 1
            
            print('save vocabulary')
            if add_unk == False:
                self.save_vocabulary('itol.csv')
            else:
                self.save_vocabulary('itow.csv')
        else:
            print('load vocabulary')
            if add_unk == False:
                self.load_vocabulary('itol.csv')
            else:
                self.load_vocabulary('itow.csv')

    def sentence_to_numeric(self, text):
        numeric_text = []

        tokenized_text = self.tokenizer(text)
        # remove stop words
        tokenized_text = compress(tokenized_text, [0 if word in self.stop_words else 1 for word in tokenized_text])

        for token in tokenized_text:
            if token in self.wtoi:
                numeric_text.append(self.wtoi[token])
            else:  # out-of-vocab words are replaced by <UNK>
                numeric_text.append(self.wtoi['<UNK>'])

        return numeric_text
