import numpy as np

path = './results/word_order/results.npz'

data = np.load(path, allow_pickle=True)

print(data['losses'])
print(data['accs'])
print(data['best_accs'])