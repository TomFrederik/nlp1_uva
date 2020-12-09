import models
import utils
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import OrderedDict

import torch.optim as optim

from nltk.treeprettyprinter import TreePrettyPrinter
from nltk import Tree



def train(config):
    #    print('config settings:')
    #    for key in config.keys():
    #        print(key,confi)
    print('Training a {} model.'.format(config.model))

    if config.use_pt_embed:
        print('Warning: Using pretrained embedding, will override given embedding dimension.')

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    
    # When running on the CuDNN backend two further options must be set for reproducibility
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # load data
    print('Loading data...')
    print('Creating subtrees!' if config.create_subtrees else '')
    train_data, dev_data, test_data = utils.get_train_test_dev(config.data_dir, create_subtrees=config.create_subtrees)

    '''
    # for testing the subtree label creation and correction
    example = dev_data[0]
    print("First example:", example)
    print("First example tokens:", example.tokens)
    print("First example label:",  example.label)
    print(TreePrettyPrinter(example.tree))
    print("First example transitions:",  example.transitions)
    print("First example subtree labels:",  example.subtree_labels)
    print(utils.get_correct_subtree_labels(example.transitions, example.subtree_labels))
    raise NotImplementedError
    '''
    ####

    # load vocabulary and embedding
    if config.use_pt_embed:
        print('Loading pretrained embedding...')

        v_pt, vectors = utils.get_pretrained_voc_vec(config.embed_path)
    
    else:
        print('Loading vocabulary from training data')
        v = utils.create_voc_from_train(train_data)

    # set sentiment dict
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})


    if config.model == 'LSTM':
        if not config.use_pt_embed:
            raise NotImplementedError

        lstm_kwargs = {
            'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1], 'hidden_dim':config.hidden_dim, 'output_dim':len(t2i)
            }

        losses, accuracies, best_accs = utils.train_loop(utils.generate_lstm, 
                                                        lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                        train_data, dev_data, test_data,
                                                        model_generator_kwargs = lstm_kwargs,
                                                        num_iterations=config.num_iterations, print_every=config.print_every, eval_every=config.eval_every, patience=config.patience,
                                                        batch_size=config.batch_size,
                                                        batch_fn=utils.get_minibatch,
                                                        prep_fn=utils.prepare_minibatch,
                                                        eval_fn=utils.evaluate,
                                                        permute=config.permute,
                                                        add_suffix=config.suffix,
                                                        result_dir=config.result_dir)
    
    elif config.model == 'TreeLSTM':
        if config.permute or not config.use_pt_embed:
            raise NotImplementedError('Permute not implemented for TreeLSTM')
        
        lstm_kwargs ={
            'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1], 'hidden_dim':config.hidden_dim, 'output_dim':len(t2i)
            }

        losses, accuracies, best_accs = utils.train_loop(utils.generate_treelstm, 
                                                        lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                        train_data, dev_data, test_data,
                                                        model_generator_kwargs = lstm_kwargs,
                                                        num_iterations=config.num_iterations, print_every=config.print_every, eval_every=config.eval_every, patience=config.patience,
                                                        batch_size=config.batch_size,
                                                        batch_fn=utils.get_minibatch,
                                                        prep_fn=utils.prepare_treelstm_minibatch,
                                                        eval_fn=utils.evaluate,
                                                        add_suffix=config.suffix,
                                                        result_dir=config.result_dir)


    elif config.model == 'BOW':
        if config.use_pt_embed or config.permute:
            raise NotImplementedError

        bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':len(t2i), 'vocab':v}

        losses, accuracies, best_accs = utils.train_loop(lambda **kwargs: models.BOW(**kwargs).to(device),
                                                            lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                            train_data, dev_data, test_data,
                                                            model_generator_kwargs = bow_kwargs,
                                                            num_iterations=config.num_iterations, print_every=config.print_every,
                                                            eval_every=config.eval_every, patience=config.patience, 
                                                            result_dir=config.result_dir)


    elif config.model == 'CBOW':
        if config.use_pt_embed or config.permute:
            raise NotImplementedError

        bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'num_classes':len(t2i)}

        losses, accuracies, best_accs = utils.train_loop(lambda **kwargs: models.CBOW(**kwargs).to(device),
                                                            lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                            train_data, dev_data, test_data,
                                                            model_generator_kwargs = bow_kwargs,
                                                            num_iterations=config.num_iterations, print_every=config.print_every,
                                                            eval_every=config.eval_every, patience=config.patience, 
                                                            result_dir=config.result_dir)                                                        

    
    elif config.model == 'DeepCBOW':
        if config.use_pt_embed or config.permute:
            raise NotImplementedError

        bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'output_dim':len(t2i), 'num_hidden':config.hidden_dim}

        losses, accuracies, best_accs = utils.train_loop(lambda **kwargs: models.DeepCBOW(**kwargs).to(device),
                                                            lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                            train_data, dev_data, test_data,
                                                            model_generator_kwargs = bow_kwargs,
                                                            num_iterations=config.num_iterations, print_every=config.print_every,
                                                            eval_every=config.eval_every, patience=config.patience, 
                                                            result_dir=config.result_dir) 


    elif config.model == 'PTDeepCBOW':
        if config.permute or not config.use_pt_embed:
            raise NotImplementedError

        bow_kwargs = {
            'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1],
            'num_hidden':config.hidden_dim, 'num_classes':len(t2i)
        }

        losses, accuracies, best_accs = utils.train_loop(utils.generate_pt_deep_cbow,
                                                            lambda model: optim.Adam(model.parameters(), lr=config.learning_rate),
                                                            train_data, dev_data, test_data,
                                                            model_generator_kwargs = bow_kwargs,
                                                            num_iterations=config.num_iterations, print_every=config.print_every,
                                                            eval_every=config.eval_every, patience=config.patience, 
                                                            result_dir=config.result_dir)       

    # save data
    np.savez_compressed(file=config.result_dir+'results.npz', losses=losses, accs=accuracies, best_accs=best_accs)
    with open(config.result_dir + 'config.json', "w") as write_file:
        json.dump(config.__dict__, write_file)
    
    # plot results
    utils.plot_results(losses, accuracies, filename=config.plot_dir+'plot.pdf', eval_every=config.eval_every,
                       print_every=config.print_every)


if __name__ == "__main__":


    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--use_pt_embed', type=bool, default=False)
    parser.add_argument('--permute', type=bool, default=False)
    parser.add_argument('--create_subtrees', type=bool, default=False)
    parser.add_argument('--embed_path', type=str, default='./googlenews.word2vec.300d.txt')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--plot_dir', type=str, default='./plots/')
    parser.add_argument('--data_dir', type=str, default='./trees/')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=150)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-4)


    config = parser.parse_args()

    # Train the model
    train(config)
