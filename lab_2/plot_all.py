import matplotlib.pyplot as plt
import numpy as np
import json
import os

models = {
    "BOW": "BOW",
    "CBOW": "CBOW",
    "DeepCBOW": "Deep CBOW",
    "PTDeepCBOW": "Deep CBOW (pretrained)",
    "LSTM": "LSTM",
    "TreeLSTM": "Tree-LSTM"
}

plt.figure(figsize=(12, 6))
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, (key, name) in enumerate(models.items()):
    path = os.path.join("results", key)

    data = np.load(os.path.join(path, 'results.npz'), allow_pickle=True)
    with open(os.path.join(path, "config.json"), "r") as read_file:
        config = json.load(read_file)

    samples_per_eval = config["eval_every"]
    # LSTM and Tree-LSTM use batches
    if key in ["LSTM", "TreeLSTM"]:
        samples_per_eval *= config["batch_size"]
    for j, accs in enumerate(data['accs']):
        plt.plot(range(0, samples_per_eval * len(accs), samples_per_eval), accs,
                 # we label only the first line of each group to avoid duplicates in the legend
                 label=(name if j == 0 else None), color=cycle[i])
plt.legend()
plt.xlim(left=1e3)
plt.xscale('log')
plt.xlabel("Samples")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("plots/all.pdf")
