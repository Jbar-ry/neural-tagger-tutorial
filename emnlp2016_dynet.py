import dynet as dy  # NN library
import random       # shuffle data
import sys          # flushing output
import numpy as np  # handle data vectors

"""
DyNet Bi-LSTM tagger tutorial
From: http://www.phontron.com/slides/emnlp2016-dynet-tutorial-part2.pdf
"""

def word_rep(w, cf_init, cb_init):
    if wc[w] > 5:    
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else: # if there are less than 5 words, e.g. rare words
        char_ids = [vc.w2i[c] for c in w]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        #concatenate the last states
        return dy.concatenate([fw_exps[-1], bw_exps[-1]]) # the -1 here just means the last char index in a word
        

WORDS_LOOKUP = model.add_lookup_parameters((nwords, 128))
CHARS_LOOKUP = model.add_lookup_parameters((nchars, 20))

cfwdRNN = dy.LSTMBuilder(1, 20, 64, model)
cbwdRNN = dy.LSTMBuilder(1, 20, 64, model)

fwdRNN = dy.LSTMBuilder(1, 128, 50, model) #128 -in dim, 50 - out dim
bwdRNN = dy.LSTMBuilder(1, 128, 50, model)

pH = model.add_parameters((32, 50*2)) # hidden layer
pO = model.add_parameters((ntags, 32)) # output layer

def build_tagging_graph(words): # important to use a function for this as we would re-use this code quite a bit
    dy.renew_cg()
    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()
    
    cf_init = cfwdRNN.initial_state()
    cb_init = cbwdRNN.initial_state()
 
    wembs = [word_rep(w, cf_init, cb_init) for w in words] 

    fws = f_init.transduce(wembs) 
    bws = b_init.transduce(reversed(wembs))

    # biLSTM states 
    bi = [dy.concatenate([f,b]) for f,b in zip(fws, reversed(bws))]

    # MLPs
    H = dy.parameter(pH)
    O = dy.parameter(pO)
    outs = [O*(dy.tanh(H*x)) for x in bi]
    return outs

def tag_sent(words):
    vecs = build_tagging_graph(words)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
        tag = np.argmax(prb) # tag is the tag w/ the highest probability
        tags.append(vt.i2w[tag]) # i2w assigns a word for the chosen index
    return zip(words, tags)

def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    losses = []
    for v, t in zip(vecs, tags):
        tid = vt.w2i[t]
        loss = dy.pickneglogsoftmax(v, tid) # cross entropy loss
        losses.append(loss)
        return dy.esum(losses) # esum is max pooling?
    
    
num_tagged = cum_loss = 0
for ITER in xrange(50):
    random.shuffle(train)
    for i, s in enumerate(train, 1):
        if i > 0 and i % 500 == 0: # print status 
            trainer.status() # need to define the trainer at some point, e.g. SGD
            print cum_loss / num_tagged
            cum_loss = num_tagged = 0
        if i % 10000 == 0: # eval on dev
            good = bad = 0.0
            for sent in dev:
                words = [w for w,t in sent]
                golds = [t for w,t in sent]
                tags = [t for w,t in tag_sent(words)]
                for go, gu in zip(golds, tags):
                    if go == gu: 
                        good +=1
                    else: 
                        bad +=1
            print good/(good+bad) # basic accuracy, e.g. number right / num total
            
            # train on sent
            words = [w for w,t in s]
            golds = [t for w,t in s]
            
            loss_exp = sent_loss(words, tags) # loss expression between words and their tags
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(golds)
            loss_exp.backward() # backpropagation
            trainer.update()
            
#fw_exps = []
#s = f_init
#for we in wembs:
#    s = s.add_input(we)
#    fw_exps.append(s.output()) 