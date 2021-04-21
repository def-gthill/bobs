"""
Functions for machine learning
"""

import pickle


def load_or_train(path, training_function):
    """
    Tries to load a pickled trained model from the specified
    path. Returns this model if the file is present. Otherwise,
    trains a new model by calling the specified training
    function (with no arguments), saves the resulting
    model to the path, and returns the model.
    """
    try:
        return load(path)
    except OSError:
        trained_model = training_function()
        save(path, trained_model)
        return trained_model


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

