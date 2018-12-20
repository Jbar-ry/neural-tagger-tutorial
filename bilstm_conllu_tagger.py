from __future__ import print_function
import time
from collections import defaultdict
import random
import math
import sys
import argparse
import dynet as dy
import numpy as np

# based off: https://github.com/neubig/nn4nlp-code/blob/ab2498eb9b25502c502202f20631cc63174da392/10-structured/bilstm-variant-tagger.py

parser = argparse.ArgumentParser(description='Tagger arguments')
parser.add_argument('--train', action='store_true', default="/home/jbarry/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu")
parser.add_argument('--dev', action='store_true', default="/home/jbarry/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu")
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
args = parser.parse_args()

train_file = args.train
dev_file = args.dev

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))


def read(filename):
    """ Parse a CoNLLU file and append words and tags to a list."""  
    with open(filename, 'r') as f:
        words, tags = [], [] # should this be after each line?
        for line in f:
            tokens = line.strip().split('\t')
            if tokens:
                if len(line.split('\t')) < 4:
                    if line.startswith('#'):
                        exit
                else:
                    word, tag = tokens[1].lower(), tokens[3]
                    words.append(w2i[word])
                    tags.append(t2i[tag])
        yield (words, tags)


# Read the data
train = list(read(train_file))
unk_word = w2i["<unk>"]
w2i = defaultdict(lambda: unk_word, w2i)
unk_tag = t2i["<unk>"]
t2i = defaultdict(lambda: unk_tag, t2i)
nwords = len(w2i)
ntags = len(t2i)
dev = list(read(dev_file))

print(nwords)
print(ntags)


# DyNet Starts
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128

# Lookup parameters for word embeddings
WORD_LOOKUP = model.add_lookup_parameters((nwords, EMBED_SIZE)) # Lookup matrix size of vocab x size of embeddings (64)


# LSTM Parameters
fwdLSTM = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model)  # Forward LSTM
bwdLSTM = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model)  # Backward LSTM


# Word-level softmax
W_sm = model.add_parameters((ntags, HIDDEN_SIZE * 2))
b_sm = model.add_parameters(ntags)


# Calculate the scores for one example
def calc_scores(words):
    dy.renew_cg()
    
    word_embs = [WORD_LOOKUP[x] for x in words]
    
    # Transduce all batch elements with an LSTM
    fwd_init = fwdLSTM.initial_state()
    fwd_word_reps = fwd_init.transduce(word_embs)
    
    bwd_init = bwdLSTM.initial_state()
    bwd_word_reps = bwd_init.transduce(reversed(word_embs))
    
    combined_word_reps = [dy.concatenate([f, b]) for f, b in zip(fwd_word_reps, reversed(bwd_word_reps))]
    
    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    scores = [dy.affine_transform([b, W, x]) for x in combined_word_reps]
    
    return scores


# Calculate MLE loss for one example
def calc_loss(scores, tags):
    losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, tags)]
    return dy.esum(losses)


# Calculate number of tags correct for one example
def calc_correct(scores, tags):
    correct = [np.argmax(score.npvalue()) == tag for score, tag in zip(scores, tags)]
    return sum(correct)


# Perform training
for ITER in range(100):
    random.shuffle(train)
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for sid in range(0, len(train)):
        this_sents += 1
        if this_sents % int(1000) == 0:
            print("train loss/word=%.4f, acc=%.2f%%, word/sec=%.4f" % (
                this_loss / this_words, 100 * this_correct / this_words, this_words / (time.time() - start)),
                  file=sys.stderr)
        # train on the example
        words, tags = train[sid]
        scores = calc_scores(words)
        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
        loss_exp.backward()
        trainer.update()
        
    # Perform evaluation 
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for words, tags in dev:
        this_sents += 1
        scores = calc_scores(words)
        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
    print("dev loss/word=%.4f, acc=%.2f%%, word/sec=%.4f" % (
        this_loss / this_words, 100 * this_correct / this_words, this_words / (time.time() - start)), file=sys.stderr)