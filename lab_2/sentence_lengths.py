import matplotlib.pyplot as plt
import numpy as np
import json
import utils
import models
from collections import OrderedDict
import torch

class namespace:
    def __init__(self, d):
        self.__dict__ = d

path = './results/'
results = np.load(path + 'results.npz', allow_pickle=True)
with open(path + "config.json", "r") as read_file:
    config = namespace(json.load(read_file))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    # When running on the CuDNN backend two further options must be set for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

train_data, dev_data, test_data = utils.get_train_test_dev(config.data_dir, create_subtrees=config.create_subtrees)
if config.use_pt_embed:
    print('Loading pretrained embedding...')

    v_pt, vectors = utils.get_pretrained_voc_vec(config.embed_path)

else:
    print('Loading vocabulary from training data')
    v = utils.create_voc_from_train(train_data)

if config.model == 'LSTM':
    if not config.use_pt_embed:
        raise NotImplementedError

    lstm_kwargs = {
        'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1], 'hidden_dim':config.hidden_dim, 'output_dim':len(t2i)
        }

    mean, std = utils.eval_models(utils.generate_lstm, test_data, lstm_kwargs,
                                  model_dir=config.result_dir)

elif config.model == 'TreeLSTM':
    if config.permute or not config.use_pt_embed:
        raise NotImplementedError('Permute not implemented for TreeLSTM')

    lstm_kwargs ={
        'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1], 'hidden_dim':config.hidden_dim, 'output_dim':len(t2i)
        }

    mean, std = utils.eval_models(utils.generate_treelstm, test_data, lstm_kwargs,
                                  model_dir=config.result_dir)


elif config.model == 'BOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':len(t2i), 'vocab':v}

    mean, std = utils.eval_models(lambda **kwargs: models.BOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir)


elif config.model == 'CBOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'num_classes':len(t2i)}

    mean, std = utils.eval_models(lambda **kwargs: models.CBOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir)


elif config.model == 'DeepCBOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'output_dim':len(t2i), 'num_hidden':config.hidden_dim}

    mean, std = utils.eval_models(lambda **kwargs: models.DeepCBOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir)


elif config.model == 'PTDeepCBOW':
    if config.permute or not config.use_pt_embed:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'output_dim':len(t2i), 'num_hidden':config.hidden_dim}

    mean, std = utils.eval_models(utils.generate_pt_deep_cbow,
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir)


print(mean, std)
