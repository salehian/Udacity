import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -np.dot(Y,np.log(P)) -np.dot(np.ones(len(Y))-Y,np.log(np.ones(len(P))-P))
