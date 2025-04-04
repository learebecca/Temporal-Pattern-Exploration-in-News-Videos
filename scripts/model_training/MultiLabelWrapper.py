class MultiLabelWrapper:
    """
    Wraps multiple single-label classifiers so that .predict() 
    returns a multi-label output array of shape (n_samples, n_labels).
    """
    def __init__(self, classifiers):
        self.classifiers = classifiers  # list of single-label estimators

    def predict(self, X):
        import numpy as np
        n_samples = X.shape[0]
        n_labels = len(self.classifiers)
        Y_pred = np.zeros((n_samples, n_labels), dtype=int)

        for i, clf in enumerate(self.classifiers):
            Y_pred[:, i] = clf.predict(X)

        return Y_pred