import numpy as np

params = [
    ("../headweights.txt", (np.load("attention_weight.npy")  / 0.0009765625).flatten()),
    ("../linearweights.txt", (np.load("linear_weight.npy") / 0.0009765625).flatten()),
    ("../linearbias.txt", (np.load("linear_bias.npy") / 1).flatten()),
    ("../ffweights1.txt", (np.load("FF_weight1.npy") / 0.001953125).flatten()),
    ("../ffweights2.txt", (np.load("FF_weight2.npy") / 0.001953125).flatten()),
    ("../ffbias2.txt", (np.load("FF_bias2.npy") / 1).flatten()),
    ("../gamma.txt", np.concatenate([np.load("gamma0.npy"), np.load("gamma1.npy")], axis=0).flatten()),
    ("../beta.txt", np.concatenate([np.load("beta0.npy"), np.load("beta1.npy")], axis=0).flatten()),
    ("../mean.txt", np.concatenate([np.load("mean0.npy"), np.load("mean1.npy")], axis=0).flatten()),
    ("../variance.txt", np.concatenate([np.load("variance0.npy"), np.load("variance1.npy")], axis=0).flatten())
]


for file, values in params:
    with open(file, "w") as f:
        for item in values:
            print(item, file=f)