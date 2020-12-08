import matplotlib.pyplot as plt
import numpy as np
import json
import utils

path = './results/word_order/'

data = np.load(path + 'results.npz', allow_pickle=True)
with open(path + "config.json", "r") as read_file:
    config = json.load(read_file)

#print(data['losses'])
#print(data['accs'])
print(data['best_accs'])
utils.plot_results(data['losses'], data['accs'], print_every=config["print_every"],
                   eval_every=config["eval_every"])
plt.show()
