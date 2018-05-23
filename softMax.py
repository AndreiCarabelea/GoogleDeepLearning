"""Softmax."""



import numpy as np
# Plot softmax curves
import matplotlib.pyplot as plt

scores = [2, 2, 2]

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis = 0);

scores  = np.array([2,2,2]);
scores = scores * 10 ;
print(scores);
print(softmax(scores));


