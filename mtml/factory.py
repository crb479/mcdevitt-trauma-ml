__doc__ = """Wrappers for quickly creating models on different data segments.

.. note:: Models used must have a ``scikit-learn`` like interface.
"""

def fit_classifier(est, X_train, y_train, cv = None, pipeline = None,
                   random_state = None, fit_params = None):
    """Factory method for producing a fitted classifier.
    
    """
    pass