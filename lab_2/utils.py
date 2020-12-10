"""
utility functions
"""

from collections import Counter, OrderedDict, defaultdict, namedtuple
import re
import math
import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from nltk import Tree
import nltk
from nltk.treeprettyprinter import TreePrettyPrinter

import models


# global variables
SHIFT = 0
REDUCE = 1

# A simple way to define a class is using namedtuple.
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path): 
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\","")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))

def examplereader(path, lower=False, create_subtrees=False):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        yield create_example(line)

        if create_subtrees:
          subtrees = get_subtrees(line)
          for tree in subtrees:
              yield create_example(tree)


def create_example(tree_string):
  tokens = tokens_from_treestring(tree_string)
  tree = Tree.fromstring(tree_string)  # use NLTK's Tree
  label = int(tree_string[1])
  trans = transitions_from_treestring(tree_string)
  return Example(tokens=tokens, tree=tree, label=label, transitions=trans)


def get_train_test_dev(dir='./trees/', lower=False, create_subtrees=False):
    train_data = list(examplereader("trees/train.txt", lower=lower, create_subtrees=create_subtrees))
    dev_data = list(examplereader("trees/dev.txt", lower=lower))
    test_data = list(examplereader("trees/test.txt", lower=lower))

    return train_data, dev_data, test_data


class OrderedCounter(Counter, OrderedDict):
  """Counter that remembers the order elements are first seen"""
  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__,
                      OrderedDict(self))
  def __reduce__(self):
    return self.__class__, (OrderedDict(self),)


class Vocabulary:
  """A vocabulary, assigns IDs to tokens"""
  
  def __init__(self):
    self.freqs = OrderedCounter()
    self.w2i = {}
    self.i2w = []

  def count_token(self, t):
    self.freqs[t] += 1
    
  def add_token(self, t):
    self.w2i[t] = len(self.w2i)
    self.i2w.append(t)    
    
  def build(self, min_freq=0):
    '''
    min_freq: minimum number of occurrences for a word to be included  
              in the vocabulary
    '''
    self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
    self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)   
    
    tok_freq = list(self.freqs.items())
    tok_freq.sort(key=lambda x: x[1], reverse=True)
    for tok, freq in tok_freq:
      if freq >= min_freq:
        self.add_token(tok)


def create_voc_from_train(train_data):
    # creates vocabulary from training data
    v = Vocabulary()
    for data_set in (train_data,):
        for ex in data_set:
                for token in ex.tokens:
                    v.count_token(token)
    
    v.build()
    
    return v


def print_parameters(model):
  total = 0
  for name, p in model.named_parameters():
    total += np.prod(p.shape)
    print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
  print("\nTotal number of parameters: {}\n".format(total))


# first prepare function
def prepare_example(example, vocab, permute=False):
  """
  Map tokens to their IDs for a single example

  Permute: whether to randomly permute word order. 
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  # vocab returns 0 if the word is not there (i2w[0] = <unk>)
  x = [vocab.w2i.get(t, 0) for t in example.tokens]
  
  x = torch.LongTensor([x])
  x = x.to(device)
  
  y = torch.LongTensor([example.label])
  y = y.to(device)
  
  
  if permute:
    idcs = np.random.permutation(np.arange(0,x.shape[1],1))    
    x = x[:,idcs]
  

  return x, y


def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
  """Accuracy of a model on given data set."""
  correct = 0
  total = 0
  model.eval()  # disable dropout (explained later)

  for example in data:
    
    if 'permute' in kwargs:
      permute = kwargs['permute']
    else:
      permute = False

    # convert the example input and label to PyTorch tensors
    x, target = prep_fn(example, model.vocab, permute)

    # forward pass without backpropagation (no_grad)
    # get the output from the neural network for input x
    with torch.no_grad():
      logits = model(x)
    
    # get the prediction
    prediction = logits.argmax(dim=-1)
    
    # add the number of correct predictions to the total correct
    correct += (prediction == target).sum().item()
    total += 1

  return correct, total, correct / float(total)




def get_examples(data, shuffle=True, **kwargs):
  """Shuffle data set and return 1 example at a time (until nothing left)"""
  if shuffle:
    print("Shuffling training data")
    random.shuffle(data)  # shuffle training data each epoch
  for example in data:
    yield example





def train_model(model, optimizer, train_data, dev_data, test_data, num_iterations=10000, 
                print_every=1000, eval_every=1000,
                batch_fn=get_examples, 
                prep_fn=prepare_example,
                eval_fn=simple_evaluate,
                permute=False,
                batch_size=1, eval_batch_size=None,
                patience=0, suffix="", result_dir='./'):
    """Train a model.
    
    The patience parameter is the number of evaluations after which
    training is stopped if the model hasn't set a new highscore (so training
    is stopped after than eval_every * patience steps without a highscore).
    If patience = 0, this feature is disabled. In any case, training is
    stopped after num_iterations. Set num_iterations to zero to treat it as infinite.
    
    Suffix is appended to the model checkpoint filename.
    Permute - whether to permute the word order
    """  
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss() # loss function
    best_eval = 0.
    best_iter = 0

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []  

    if eval_batch_size is None:
        eval_batch_size = batch_size
  
    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):

            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)

            B = targets.size(0)  # later we will use B examples per update

            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()

            # backward pass (tip: check the Introduction to PyTorch notebook)

            # erase previous gradients
            optimizer.zero_grad()
            # YOUR CODE HERE

            # compute gradients
            # YOUR CODE HERE
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            # YOUR CODE HERE
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                print("Iter %r: loss=%.4f, time=%.2fs" % 
                      (iter_i, train_loss, time.time()-start))
                losses.append(train_loss)
                print_num = 0        
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size,
                                         batch_fn=batch_fn, prep_fn=prep_fn, permute=permute)
                accuracies.append(accuracy)
                print("iter %r: dev acc=%.4f" % (iter_i, accuracy))       
        
                # save best model parameters
                if accuracy > best_eval:
                    print("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = result_dir + "{}_{}.pt".format(model.__class__.__name__, suffix)
                    ckpt = {
                      "state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "best_eval": best_eval,
                      "best_iter": best_iter
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations or ((patience > 0) and iter_i - best_iter >= patience * eval_every):
                if patience > 0 and (iter_i - best_iter >= patience * eval_every):
                    print("Stopping early because there was no improvement for {} steps".format(iter_i - best_iter))
                print("Done training")

                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = result_dir + "{}_{}.pt".format(model.__class__.__name__, suffix)        
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])

                _, _, train_acc = eval_fn(
                    model, train_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, dev_acc = eval_fn(
                    model, dev_data, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, test_acc = eval_fn(
                    model, test_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)

                print("best model iter {:d}: "
                      "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                          best_iter, train_acc, dev_acc, test_acc))

                return (losses, accuracies), (train_acc, dev_acc, test_acc)

def train_loop(model_generator, optimizer_generator, train_data, dev_data, test_data,
               model_generator_kwargs = {},
               num_iterations=10000, 
               print_every=1000, eval_every=1000,
               batch_fn=get_examples, 
               prep_fn=prepare_example,
               eval_fn=simple_evaluate,
               permute=False,
               batch_size=1, eval_batch_size=None,
               patience=0,
               num_seeds=3,
               add_suffix='',
               result_dir='./'):
    """Train a model with multiple seeds.
    
    model_generator should be a function that returns a model instance.
    optimizer_generator should be a function that takes a model instance
    and return an optimizer.
    
    Returns three lists: losses, accuracies, best_accuracies.
    For the first two, the first axis enumerates the different seeds and
    the second axis is time. For best_accuracies, the first axis has length
    three and contains the train, validation and test accuracies (in that order).
    The second axis enumerates seeds."""
    print('Permute is turned on!' if permute else 'Permute is turned off.')
    losses = []
    accuracies = []
    best_accuracies = []
    for seed in range(num_seeds):
        print("Training new model with seed {}".format(seed))
        print("=" * 50)

        ## Seeding
        # seed torch
        torch.manual_seed(seed)
        
        # seed cuda
        torch.cuda.manual_seed_all(seed)
        
        # seed random        
        random.seed(seed)

        # seed numpy
        np.random.seed(seed)

        ##
        
        
        # set up model and optimizer
        model = model_generator(**model_generator_kwargs)
        optimizer = optimizer_generator(model)

        # train model
        (new_losses, new_accuracies), best_accs = train_model(model, optimizer, train_data, dev_data, test_data, num_iterations,
                    print_every, eval_every, batch_fn, prep_fn, eval_fn, permute,
                    batch_size, eval_batch_size, patience, suffix=add_suffix+str(seed), result_dir=result_dir)
        
        # log performance
        losses.append(new_losses)
        accuracies.append(new_accuracies)
        best_accuracies.append(best_accs)
    
    # transpose accuracy list
    best_accuracies = list(zip(*best_accuracies))

    print("Test accuracy:", np.mean(best_accuracies[2]), "+-", np.std(best_accuracies[2]))
    return losses, accuracies, best_accuracies


def plot_results(losses, accuracies, filename=None, eval_every=100, print_every=100):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for run in losses:
        plt.plot(range(0, print_every * len(run), print_every), run)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    for run in accuracies:
        plt.plot(range(0, eval_every * len(run), eval_every), run)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


def get_pretrained_voc_vec(path):
    # gets vocabulary and embedding vectors from a pretrained
    # word embedding located at path
    vectors = [[0]*300, [0]*300] # first two vectors are embedding of <unk> and <pad>

    f = open(path)
    v_pt = Vocabulary()
    for line in f:
        line_list = line.split()
        word = line_list[0]
        vec = line_list[1:]
        vec = [float(val) for val in vec]
        vectors.append(vec)
        v_pt.count_token(word)
        
    vectors = np.stack(vectors, axis=0)
    v_pt.build()

    return v_pt, vectors


def generate_pt_deep_cbow(v_pt, embed_vectors, embed_dim, num_classes, num_hidden):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pt_deep_cbow_model = models.PTDeepCBOW(len(v_pt.i2w), embed_dim, num_classes, num_hidden, v_pt)

    # copy pre-trained word vectors into embeddings table
    pt_deep_cbow_model.embed.weight.data.copy_(torch.from_numpy(embed_vectors))

    # disable training the pre-trained embeddings
    pt_deep_cbow_model.embed.weight.requires_grad = False

    # move model to specified device
    pt_deep_cbow_model = pt_deep_cbow_model.to(device)
    return pt_deep_cbow_model


def generate_lstm(v_pt, embed_vectors, embed_dim, hidden_dim, output_dim):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lstm_model = models.LSTMClassifier(len(v_pt.w2i), embed_dim, hidden_dim, output_dim, v_pt)

    # copy pre-trained word vectors into embeddings table
    with torch.no_grad():
        lstm_model.embed.weight.data.copy_(torch.from_numpy(embed_vectors))
        lstm_model.embed.weight.requires_grad = False

    lstm_model = lstm_model.to(device)
    return lstm_model

def generate_treelstm(v_pt, embed_vectors, embed_dim, hidden_dim, output_dim):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tree_model = models.TreeLSTMClassifier(len(v_pt.w2i), embed_dim, hidden_dim, output_dim, v_pt)

    # copy pre-trained word vectors into embeddings table
    with torch.no_grad():
        tree_model.embed.weight.data.copy_(torch.from_numpy(embed_vectors))
        tree_model.embed.weight.requires_grad = False

    tree_model = tree_model.to(device)
    return tree_model


def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""
  
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
  
    batch = []
  
    # yield minibatches
    for example in data:
        batch.append(example)
    
        if len(batch) == batch_size:
            yield batch
            batch = []
      
    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, permute=False):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.

    Permute: whether to permute tokens
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    if permute: # permute token order before padding
      # vocab returns 0 if the word is not there
      x = [pad(list(np.random.permutation([vocab.w2i.get(t, 0) for t in ex.tokens])), maxlen) for ex in mb]
    else:
      # vocab returns 0 if the word is not there
      x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y


def evaluate(model, data, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch, permute=False,
             batch_size=16):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab, permute)
        with torch.no_grad():
            logits = model(x)
      
        predictions = logits.argmax(dim=-1).view(-1)
    
        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)



def batch(states):
  """
  Turns a list of states into a single tensor for fast processing. 
  This function also chunks (splits) each state into a (h, c) pair"""
  return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
  """
  Turns a tensor back into a list of states.
  First, (h, c) are merged into a single state.
  Then the result is split into a list of sentences.
  """
  return torch.split(torch.cat(state, 1), 1, 0)


def prepare_treelstm_minibatch(mb, vocab, permute=False):
  """
  Returns sentences reversed (last word first)
  Returns transitions together with the sentences.  
  """
  if permute:
      raise NotImplementedError('Permute not implemented for TreeLSTM')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])
    
  # vocab returns 0 if the word is not there
  # NOTE: reversed sequence!
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]
  
  x = torch.LongTensor(x)
  x = x.to(device)
  
  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)
  
  maxlen_t = max([len(ex.transitions) for ex in mb])
  transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
  transitions = np.array(transitions)
  transitions = transitions.T  # time-major
  
  return (x, transitions), y


def sentence_length_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """Accuracy of a model for different sentence lengths."""
    correct = defaultdict(int)
    total = defaultdict(int)
    model.eval()  # disable dropout (explained later)

    for example in data:
        
        # convert the example input and label to PyTorch tensors
        x, target = prep_fn(example, model.vocab)

        # forward pass without backpropagation (no_grad)
        # get the output from the neural network for input x
        with torch.no_grad():
            logits = model(x)

        # get the prediction
        prediction = logits.argmax(dim=-1)

        # add the number of correct predictions to the total correct
        length = len(example.tokens)
        correct[length] += (prediction == target).int().item()
        total[length] += 1

    accuracy = {length: correct[length] / float(total[length]) for length in correct}
    max_length = max(accuracy.keys())
    # Turn it into a masked numpy array so we can later take mean etc.
    result = np.ma.array(
        list(accuracy[k] if k in accuracy else np.ma.masked
             for k in range(max_length + 1)))

    return correct, total, result



def eval_models(model_generator, test_data,
                model_generator_kwargs={},
                model_dir="results/",
                batch_fn=get_examples, 
                prep_fn=prepare_example,
                eval_fn=sentence_length_evaluate,
                eval_batch_size=1,
                device=None):
    accuracies = []
    for seed in range(3):
        model = model_generator(**model_generator_kwargs)
        path = model_dir + "{}_{}.pt".format(model.__class__.__name__, seed)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        #_, _, train_acc = eval_fn(
        #    model, train_data, batch_size=eval_batch_size, 
        #    batch_fn=batch_fn, prep_fn=prep_fn)
        #_, _, dev_acc = eval_fn(
        #    model, dev_data, batch_size=eval_batch_size,
        #    batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, test_acc = eval_fn(
            model, test_data, batch_size=eval_batch_size, 
            batch_fn=batch_fn, prep_fn=prep_fn)
        accuracies.append(test_acc)
    accuracies = np.stack(accuracies)
    return np.mean(accuracies, axis=0), np.std(accuracies, axis=0)



    
def get_subtrees(treestring):
  '''
  given a raw treestring, will output all subtree strings
  will not produce single word subtrees
  '''
  # first remove top level tree (k ... )
  treestring = treestring[2:-1]

  # find positions of "("
  opening_idcs = [i for i, letter in enumerate(treestring) if letter=='(']

  
  # find closing positions for each opening position
  # ignore if single token
  subtrees = []
  for idx in opening_idcs:
    num_descendents = 0
    ctr = 1
    for j, letter in enumerate(treestring[idx+1:]):
      if letter == ')':
        ctr -= 1
        if ctr == 0:
          closing_idx = j + idx + 1
          break
        elif ctr > 0:
          num_descendents += 1
      elif letter == '(':
        ctr += 1

    if num_descendents > 0:
      subtrees.append(treestring[idx:closing_idx+1])
  
  return subtrees
