"""
Extract Features from Provided Invariants
"""
import numpy as np

def getFeatureSet1(sij, rij):
    # If Sij is 5D, then assume nX x nY x nZ
    if len(sij.shape) == 5:
    # If Sij is 4D, then assume nX x nY x 3 x 3
    if len(sij.shape) == 4:
        sij = sij.reshape((sij.shape[0]*sij.shape[1], 9))

    # Feature set 1 invariants is nPoint x 6 features
    invariants = np.zeros((sij.shape[0], 6))


# def


