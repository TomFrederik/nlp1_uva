"""
Only needed to evaluate models with word shuffling that were trained without shuffling.
To train with shuffling, use the permute flag for train.py
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import utils
import models
from collections import OrderedDict
import argparse
import os
import torch

class namespace:
    def __init__(self, d):
        self.__dict__ = d

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the results directory")
args = parser.parse_args()

results = np.load(os.path.join(args.path, 'results.npz'), allow_pickle=True)
with open(os.path.join(args.path, "config.json"), "r") as read_file:
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
                                  batch_fn=utils.get_minibatch,
                                  prep_fn=utils.prepare_minibatch,
                                  eval_fn=utils.evaluate,
                                  eval_batch_size=16,
                                  eval_kwargs={"permute": True},
                                  model_dir=config.result_dir, device=torch.device(device))
else:
    raise NotImplementedError("Shuffling words only supported for LSTM")


print("Accuracy:", mean, "+-", std)
