import pdb
import matplotlib.pyplot as plt
import numpy as np
def plot():
    implicit_loss = np.load("adj_vs_implicit_final/naphthalene_100tau/implicit_RDFloss.npy")
    plt.plot(range(len(implicit_loss)), implicit_loss, label = 'implicit loss')
    adjoint_loss = np.load("adj_vs_implicit_final/naphthalene_100tau/adjointRDFloss.npy")
    plt.plot(range(len(adjoint_loss)), adjoint_loss, label='adjoint loss' )
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("adj_vs_implicit_final/naphthalene_100tau/loss.jpg")

plot()