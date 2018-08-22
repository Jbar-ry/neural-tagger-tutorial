import argparse     # process command line arguments
import random       # shuffle data
import sys          # flushing output
import numpy as np  # handle data vectors

#import dynet_config
#dynet_config.set(mem=256, autobatch=0, weight_decay=WEIGHT_DECAY,random_seed=0)
# dynet_config.set_gpu() for when we want to run with GPUs

# I am trying to encapsulate the params in a class and make other functions accordingly.
# I am mainly following https://github.com/IndicoDataSolutions/spaCy/blob/master/examples/spacy_dynet_lstm.py as a tutorial on how to do this.
# *Work in Progress*


import dynet as dy 
    
PAD = "__PAD__"
UNK = "__UNK__"

def main():
    parser = argparse.ArgumentParser(description='POS tagger, default values match Jiang, Liang and Zhang (CoLing 2018).')    
    parser.add_argument("--trainfile", metavar="FILE", help="Annotated CONLL(U) train file")
    parser.add_argument("--devfile", metavar="FILE", help="Annotated CONLL(U) dev file")
    parser.add_argument("--testfile", metavar="FILE", help="Annotated CONLL(U) test file")
    parser.add_argument("--EXTERNAL_EMBEDDING", metavar="FILE", help="location of glove vectors", default="../data/glove.6B.100d.txt")
    parser.add_argument("--DIM_EMBEDDING", help="number of dimensions in our word embeddings", required=False, default=100)
    parser.add_argument("--LSTM_HIDDEN", help="number of dimensions in the hidden vectors for the LSTM. Based on NCRFpp (200 in the paper, but 100 per direction in code)", required=False,type=int, default=100)
    parser.add_argument("--BATCH_SIZE", help="number of examples considered in each model update.", required=False, type=int, default=10)
    parser.add_argument("--LEARNING_RATE", help="adjusts how rapidly model parameters change by rescaling the gradient vector.", required=False, type=int, default=0.015)
    parser.add_argument("--LEARNING_DECAY_RATE", help="part of a rescaling of the learning rate after each pass through the data.", required=False, type=int, default=0.05)
    parser.add_argument("--EPOCHS", help="number of passes through the data in training.", required=False, type=int, default=100)
    parser.add_argument("--KEEP_PROB", help="probability of keeping a value when applying dropout.", required=False, type=int, default=0.5)
    parser.add_argument("--WEIGHT_DECAY", help="part of a rescaling of weights when an update occurs.", required=False, type=int, default=1e-8)

    args = parser.parse_args()
    
    train = read_conllu_file(args.trainfile)
    dev = read_conllu_file(args.devfile)
    
    tokens, tags, w2i, t2i, NWORDS, NTAGS = build_vocab(train, dev)
    
    tagger = Bi_Tagger(tokens, tags, w2i, t2i, args)
    
    # training loop/ evaluation
#    loss = 0
#    match = 0
#    total = 0
#    start = 0    
#    for iter in range(100): # change this later to epochs from command line
#        random.shuffle(train)
#        while start < len(train):
#            batch = data[start: start + args.BATCH_SIZE]
#            batch.sort(key = lambda x: -len(x[0]))
#            start += args.BATCH_SIZE
#            if start % 4000 == 0:
#                tagger.trainer.status()
#                print(loss, match / total)
#                sys.stdout.flush()        
        
        
        
        
# combine the losses for the batch, do an update, and record the loss

#loss_for_batch = dy.esum(loss_expressions)
#loss_for_batch.backward()
#trainer.update()
#loss += loss_for_batch.scalar_value()
#
#####
## Update the number of correct tags and total tags
#for (_, g), a in zip(batch, predicted):
#total += len(g)
#for gt, at in zip(g, a):
#gt = t2i[gt]
#if gt == at:
#match += 1
#
#return loss, match / total        
    
def build_vocab(train, dev):    
    ### Build the dictionaries: Word2indices (w2i) and tag2indices (t2i).
    ### Each word and tag will have a dictionary mapping from the string to an id/index which will be fed into the model. 
    ### We also construct inverse dictionaries i2w/t for outputting word/tag predictions.
    ### We need to train good representations for UNK (unknown words) as we will encounter many UNK words at test time.
    i2w = [PAD, UNK] 
    w2i = {PAD: 0, UNK: 1} # word to index with values for padding and unknown/OOV tokens
    i2t = [PAD]
    t2i = {PAD: 0} 
        
    ### Iterate through tokens and tags and assign the next integer id for words and tags which we have not seen yet in our dictionaries.
    for tokens, tags in train + dev:
        for token in tokens:
            token = simplify_token(token)
            if token not in w2i:
                w2i[token] = len(w2i) 
                i2w.append(token)
        for tag in tags:
            if tag not in t2i:
                t2i[tag] = len(t2i)
                i2t.append(tag)      
        NWORDS = len(w2i)
        NTAGS = len(t2i)
        return(tokens, tags, w2i, t2i, NWORDS, NTAGS)
  
      
### Encapsulate the parameters in a class, as per Yoav's suggestion.                
class Bi_Tagger(object):
    
    def __init__(self, tokens, tags, DIM_EMBEDDING, LSTM_HIDDEN, BATCH_SIZE, LEARNING_RATE, \
                LEARNING_DECAY_RATE, EPOCHS, KEEP_PROB, WEIGHT_DECAY, w2i, t2i, NWORDS, NTAGS, pretrained_list, model, args):
        self.model = dy.ParameterCollection()
        random.seed(1)
        
        # adding lots of params for now
        self.tokens = tokens
        self.tags = tags
        self.w2i = w2i
        self.t2i = t2i
        self.NWORDS = len(w2i)
        self.NTAGS = len(t2i)
        self.tokens_batch = []
        self.tags_batch = []
        
        # dimension sizes
        self.DIM_EMBEDDING = args.DIM_EMBEDDING
        self.LSTM_HIDDEN = args.LSTM_HIDDEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.LEARNING_RATE = args.LEARNING_RATE
        self.LEARNING_DECAY_RATE = args.LEARNING_DECAY_RATE
        self.EPOCHS = args.EPOCHS
        self.KEEP_PROB = args.KEEP_PROB
        self.WEIGHT_DECAY = args.WEIGHT_DECAY
        self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=self.LEARNING_RATE) # updates model
        ### DyNet clips gradients by default, which we disable here (this can have a big impact on performance).
        #trainer.set_clip_threshold(-1)

        ### Model creation
        ### Create word embeddings and initialise
        ### Lookup parameters are a matrix that supports efficient sparse lookup.
        self.pEmbedding = self.model.add_lookup_parameters((self.NWORDS, self.DIM_EMBEDDING)) # pre-trained embeddings from GloVe
        self.pEmbedding.init_from_array(np.array(self.pretrained_list))
        
        
        # Create LSTM parameters
        #### Objects that create LSTM cells and the necessary parameters.
        stdv = 1.0 / np.sqrt(self.LSTM_HIDDEN) # Needed to match PyTorch
        self.f_lstm = dy.VanillaLSTMBuilder(1, self.DIM_EMBEDDING, self.LSTM_HIDDEN, self.model,
                forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
        self.b_lstm = dy.VanillaLSTMBuilder(1, self.DIM_EMBEDDING, self.LSTM_HIDDEN, self.model,
                forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
        
        # Create output layer
        self.pOutput = self.model.add_parameters((self.NTAGS, 2 * self.LSTM_HIDDEN))
        
        # Set recurrent dropout values (not used in this case)
        self.f_lstm.set_dropouts(0.0, 0.0)
        self.b_lstm.set_dropouts(0.0, 0.0)
        # Initialise LSTM parameters
        #### To match PyTorch, we initialise the parameters with an unconventional approach.
        self.f_lstm.get_parameters()[0][0].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN, self.DIM_EMBEDDING]))
        self.f_lstm.get_parameters()[0][1].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN, self.LSTM_HIDDEN]))
        self.f_lstm.get_parameters()[0][2].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN]))
        self.b_lstm.get_parameters()[0][0].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN, self.DIM_EMBEDDING]))
        self.b_lstm.get_parameters()[0][1].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN, self.LSTM_HIDDEN]))
        self.b_lstm.get_parameters()[0][2].set_value(
                np.random.uniform(-stdv, stdv, [4 * self.LSTM_HIDDEN]))
        
        
    ### Build computation graph and return expression
    def __call__(self, tokens, tags):
        dy.renew_cg()
        
        ### Convert tokens and tags from strings to numbers using the indices.                   
        token_ids = [self.w2i.get(simplify_token(t), 0) for t in tokens] 
        tag_ids = [self.t2i[t] for t in tags]
    
        wembs = [dy.lookup(self.pEmbedding, w) for w in token_ids]
        rev_embs = reversed(wembs)
        
        f_init = self.f_lstm.initial_state()
        b_init = self.b_lstm.initial_state()
        
        f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]  # take the output vectors from the cell state at each step. 
        b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]
    
        # For each output, calculate the output and loss
        pred_tags=[]
        for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
            # Combine the outputs.
            combined = dy.concatenate([f,b])
            # Apply dropout
            if train:
                wembs = [dy.dropout(w, 1.0 - self.KEEP_PROB) for w in wembs]
            # Matrix multiply to get scores for each tag
            r_t = self.pOutput * combined            
            # Calculate cross-entropy loss
            if train:
                err = dy.pickneglogsoftmax(r_t, t)
                #### We are not actually evaluating the loss values here, instead we collect them together in a list. This enables DyNet's <a href="http://dynet.readthedocs.io/en/latest/tutorials_notebooks/Autobatching.html">autobatching</a>.
                loss_expressions.append(err) # this needs to be created elsewhere.
            # Calculate the highest scoring tag
            #### This call to .npvalue() will lead to evaluation of the graph and so we don't actually get the benefits of autobatching. With some refactoring we could get the benefit back (simply keep the r_t expressions around and do this after the update), but that would have complicated this code.
            chosen = np.argmax(r_t.npvalue())
            pred_tags.append(chosen)
            return(pred_tags)
            
            #return expression
   
            # alternatively, could try:
            # out = dy.softmax(r_t)
            # pred_tags.append(self.i2w[np.argmax(chosen,npvalue())])
                                  
    # similar functions to spacy's
    def predict_batch(self, words_batch):
        dy.renew_cg()
        length = max(len(tokens) for tokens in self.BATCH_SIZE)
        
        
    def pipe(self, sentences):
        batch = []
        for words in sentences:
            batch.append(words)
            if len(batch) == self.BATCH_SIZE:
                tags_batch = self.predict_batch(batch)
                for words, tags in zip(batch, tags_batch):
                    yield tags
                batch = []
                
    def update(self, words, tags):
        self.tokens_batch.append(words)
        self.tags_batch.append(tags)
        if len(self.tokens_batch) == self.BATCH_SIZE:
            loss = self.update_batch(self.tokens_batch, self.tags_batch)
            self.tokens_batch = []
            self.tags_batch = []
        else:
            loss = 0
        return(loss)
            
            
    def update_batch(self, batch): # try just add in one batch argument and separate the words and tags after.
        dy.renew_cg()
        ### Convert tokens and tags from strings to numbers using the indices.
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [self.w2i.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [self.t2i[t] for t in tags]
            
        wembs = [dy.lookup(self.pEmbedding, w) for w in token_ids]
        rev_embs = reversed(wembs)
        
        f_init = self.f_lstm.initial_state()
        b_init = self.b_lstm.initial_state()
        
        f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]  # take the output vectors from the cell state at each step. 
        b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]
        
        
        errs = []
        for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
            # Combine the outputs.
            combined = dy.concatenate([f,b])
            r_t = self.pOutput * combined  
            err = dy.pickneglogsoftmax_batch(r_t, tag_ids[i])
            errs.append(dy.sum_batches(err))
        sum_errs = dynet.esum(errs)
        squared = -sum_errs # * sum_errs
        losses = sum_errs.scalar_value()
        sum_errs.backward()
        self.trainer.update()
        return losses

        
        
        
        
#                if len(loss) == 0:
#            self._init_cg(train=True)
#            return 0.
#
#        loss = esum(loss) * (1. / self._batch_size)
#        ret = loss.scalar_value()
#        loss.backward()
#        self._trainer.update()
#        self._steps += 1
#        self._init_cg(train=True)
    
         
        
        #### To make the code match across the three versions, we group together some framework specific values needed when doing a pass over the data.
        expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)
        
        #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
        for epoch in range(self.EPOCHS):
            random.shuffle(train)
    
            ####
            # Update learning rate
            trainer.learning_rate = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)
    
            #### Training pass.
            loss, train_acc = do_pass(train, w2i, t2i, expressions, True,
                    current_lr)
            #### Dev pass.
            _, dev_acc = do_pass(dev, w2i, t2i, expressions, False)
            print("{} loss {} train-acc {} dev-acc {}".format(epoch, loss, train_acc, dev_acc))
    
        #### The syntax varies, but in all three cases either saving or loading the parameters of a model must be done after the model is defined.
        # Save model
        model.save("tagger.dy.model")
    
        # Load model
        model.populate("tagger.dy.model")
    
        # Evaluation pass.
        _, test_acc = do_pass(dev, w2i, t2i, expressions, False)
        print("Test Accuracy: {:.3f}".format(test_acc))



#    def evaluate(self, train, predicted_tags):
#        """
#        Compute accuraccy on a dev/test file
#        """
#        correct = 0
#        total = 0.0
#        
#        if predicted_tags != None:
            
        
    #### Inference (the same function for train and test).
    def do_pass(data, w2i, t2i, expressions, train):
        pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions
    
        # Loop over batches
        loss = 0
        match = 0
        total = 0
        start = 0
        while start < len(data):
            #### Form the batch and order it based on length (important for efficient processing in PyTorch).
            batch = data[start : start + BATCH_SIZE]
            batch.sort(key = lambda x: -len(x[0]))
            start += BATCH_SIZE
#            self._words_batch = []
#            self._tags_batch = []
#            self._minibatch_size = 32    
            #### Log partial results so we can conveniently check progress.
            if start % 4000 == 0:
                print(loss, match / total)
                sys.stdout.flush()
    
            #### Start a new computation graph for this batch.
            # Process batch
            dy.renew_cg()
            #### For each example, we will construct an expression that gives the loss.
            loss_expressions = []
            predicted = []
            #### Convert tokens and tags from strings to numbers using the indices.
            for n, (tokens, tags) in enumerate(batch):
                token_ids = [w2i.get(simplify_token(t), 0) for t in tokens]
                tag_ids = [t2i[t] for t in tags]
    
                #### Now we define the computation to be performed with the model. Note that they are not applied yet, we are simply building the computation graph.
                # Look up word embeddings
                wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
                # Apply dropout
                if train:
                    wembs = [dy.dropout(w, 1.0 - KEEP_PROB) for w in wembs]
                # Feed words into the LSTM
                #### Create an expression for two LSTMs and feed in the embeddings (reversed in one case).
                #### We pull out the output vector from the cell state at each step.
                f_init = f_lstm.initial_state()
                f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
                rev_embs = reversed(wembs)
                b_init = b_lstm.initial_state()
                b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]
    
                # For each output, calculate the output and loss
                pred_tags = []
                for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
                    # Combine the outputs
                    combined = dy.concatenate([f,b])
                    # Apply dropout
                    if train:
                        combined = dy.dropout(combined, 1.0 - KEEP_PROB)
                    # Matrix multiply to get scores for each tag
                    r_t = pOutput * combined
                    # Calculate cross-entropy loss
                    if train:
                        err = dy.pickneglogsoftmax(r_t, t)
                        #### We are not actually evaluating the loss values here, instead we collect them together in a list. This enables DyNet's <a href="http://dynet.readthedocs.io/en/latest/tutorials_notebooks/Autobatching.html">autobatching</a>.
                        loss_expressions.append(err)
                    # Calculate the highest scoring tag
                    #### This call to .npvalue() will lead to evaluation of the graph and so we don't actually get the benefits of autobatching. With some refactoring we could get the benefit back (simply keep the r_t expressions around and do this after the update), but that would have complicated this code.
                    chosen = np.argmax(r_t.npvalue())
                    pred_tags.append(chosen)
                predicted.append(pred_tags)
    
            # combine the losses for the batch, do an update, and record the loss
            if train:
                loss_for_batch = dy.esum(loss_expressions)
                loss_for_batch.backward()
                trainer.update()
                loss += loss_for_batch.scalar_value()
    
            ####
            # Update the number of correct tags and total tags
            for (_, g), a in zip(batch, predicted):
                total += len(g)
                for gt, at in zip(g, a):
                    gt = t2i[gt]
                    if gt == at:
                        match += 1
    
        return loss, match / total 
        #match / total is a generic accuracy metric that is used for train, dev, and test cases.        
        
        def get_external_embeddings(self, DIM_EMBEDDING, external_embedding_file):
            ### Load pre-trained GloVe vectors (100-d)
            pretrained = {}
            for line in open(external_embedding_file):
                parts = line.strip().split()
                word = parts[0]
                vector = [float(v) for v in parts[1:]]
                pretrained[word] = vector # set mappings of word to the vector
                ### We need the word vectors as a list to initialise the embeddings.
                ### Each entry in the list corresponds to the token with that index.
                pretrained_list = []
                scale = np.sqrt(3.0 / DIM_EMBEDDING)
                for word in i2w:
                    # apply lower() because all GloVe vectors are for lowercase words
                    if word.lower() in pretrained:
                        pretrained_list.append(np.array(pretrained[word.lower()]))
                    else:
                        #### For words that do not appear in GloVe we generate a random vector (note, the choice of scale here is important and we follow Jiang, Liang and Zhang (CoLing 2018).
                        random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
                        pretrained_list.append(random_vector)
                return(pretrained_list)
                
#        self.EXTERNAL_EMBEDDING = None
#        if args.EXT_EMBEDDING is not None:
#            self.get_external_embeddings(args.EXTERNAL_EMBEDDING)
#            GLOVE = "../data/glove.6B.100d.txt" # location of glove vectors
#            print("USING GLOVE EMBEDDINGS FROM {} ".format(GLOVE))


def read_conllu_file(filename): # based off: https://github.com/bplank/bilstm-aux/blob/master/src/lib/mio.py
    print('loading: ' + filename)
    current_words = []
    current_tags = []
    for line in open(filename):
        line = line.strip()   
        if line:
            if len(line.split('\t')) < 4: # metadata, e.g. lines without the 10 conllu columns
                if line.startswith('#'): # skip comments
                    exit
            else:
                word, tag = line.split('\t')[1], line.split('\t')[3]
                current_words.append(word)
                current_tags.append(tag)
        else:
            if current_words: # skip emtpy lines
                yield (current_words, current_tags)
            current_words = []
            current_tags = []
    # check for last one
    if current_tags != []:
        yield (current_words, current_tags)
        
def simplify_token(token):
    chars = []
    for char in token:
        #### Reduce sparsity by replacing all digits with 0.
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

if __name__ == '__main__':
    main()  