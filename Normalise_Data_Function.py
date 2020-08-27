import numpy as np

def normalise(X, normalisation_type = "standardisation"):
    
    normalised_X = np.zeros_like(X)

    if normalisation_type == "standardisation":  # perform standardisation/z normalisation
        # X - mean/standard deviation
        means = np.mean(X, axis=0)
        sds = np.std(X, axis=0)
        normalised_X = (X - means)/sds
        return normalised_X
    
    elif normalisation_type == "min_max":     # perform min-max normalisation
        # X - Min(x)/Range
        minimum = X.min(axis=0)
        maximum = X.max(axis=0)
        normalised_X = (X - minimum)/(maximum - minimum)
        return normalised_X

    elif normalisation_type == "unit_vector":     # perform unit-vector normalisation
        # X/|X|
        normalised_X = X.T/np.linalg.norm(X, axis=1)
        return normalised_X.T
    else:
        return X

    