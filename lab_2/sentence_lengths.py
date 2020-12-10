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
parser.add_argument("--plot_name", type=str, required=True, help="Filename for the plot (without extension)")
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
                                  model_dir=config.result_dir, device=torch.device(device))

elif config.model == 'TreeLSTM':
    if config.permute or not config.use_pt_embed:
        raise NotImplementedError('Permute not implemented for TreeLSTM')

    lstm_kwargs ={
        'v_pt':v_pt, 'embed_vectors':vectors, 'embed_dim':vectors.shape[1], 'hidden_dim':config.hidden_dim, 'output_dim':len(t2i)
        }

    mean, std = utils.eval_models(utils.generate_treelstm, test_data, lstm_kwargs,
                                  batch_fn=utils.get_minibatch,
                                  prep_fn=utils.prepare_treelstm_minibatch,
                                  eval_fn=utils.sentence_length_batch_evaluate,
                                  model_dir=config.result_dir, device=torch.device(device))


elif config.model == 'BOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':len(t2i), 'vocab':v}

    mean, std = utils.eval_models(lambda **kwargs: models.BOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir, device=torch.device(device))


elif config.model == 'CBOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'num_classes':len(t2i)}

    mean, std = utils.eval_models(lambda **kwargs: models.CBOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir, device=torch.device(device))


elif config.model == 'DeepCBOW':
    if config.use_pt_embed or config.permute:
        raise NotImplementedError

    bow_kwargs = {'vocab_size':len(v.w2i), 'embedding_dim':config.embed_dim, 'vocab':v, 'output_dim':len(t2i), 'num_hidden':config.hidden_dim}

    mean, std = utils.eval_models(lambda **kwargs: models.DeepCBOW(**kwargs).to(device),
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir, device=torch.device(device))


elif config.model == 'PTDeepCBOW':
    if config.permute or not config.use_pt_embed:
        raise NotImplementedError

    bow_kwargs = {'embed_vectors': vectors, 'embed_dim': vectors.shape[1], 'v_pt':v_pt, 'num_classes':len(t2i), 'num_hidden':config.hidden_dim}

    mean, std = utils.eval_models(utils.generate_pt_deep_cbow,
                                  test_data, bow_kwargs,
                                  model_dir=config.result_dir, device=torch.device(device))


print(mean, std)

plt.plot(mean)
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
plt.xlabel("Sentence length")
plt.ylabel("Accuracy")
os.makedirs("plots/sentence_lengths", exist_ok=True)
plt.savefig(os.path.join("plots/sentence_lengths", args.plot_name + ".pdf"))
