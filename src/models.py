"""The three models from the paper.

1. RandomBaseline           - guesses 0/1 at random (the "floor")
2. NewtonLogisticRegression - logistic regression solved with Newton's method,
                              written from scratch in numpy (paper Section 4)
3. build_nn                 - 3 hidden layers x 256 units, ReLU, sigmoid output,
                              Adam lr=0.001 (paper Section 5.2)
"""
import numpy as np
from sklearn.neural_network import MLPClassifier


class RandomBaseline:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.rng.integers(0, 2, size=len(X))


class NewtonLogisticRegression:
    """Logistic regression trained with Newton's method.

    Update rule:  theta <- theta - H^-1 * grad
      grad = X^T (sigmoid(X theta) - y) / m      (+ l2 * theta)
      H    = X^T diag(p(1-p)) X / m              (+ l2 * I)
    A small L2 term keeps H invertible (pure Newton can blow up when
    the data is separable or features are collinear - this is likely
    why the paper's AMC model "never converged").
    """

    def __init__(self, max_iter=20, tol=1e-6, l2=1e-4, class_weight=None, penalize_bias=True):
        # class_weight="balanced" weights each class by n / (2 * n_class), which helps when
        # one class is much rarer. penalize_bias=False leaves the intercept out of the L2
        # term (like scikit-learn). Defaults reproduce the original behaviour exactly.
        self.max_iter, self.tol, self.l2 = max_iter, tol, l2
        self.class_weight, self.penalize_bias = class_weight, penalize_bias
        self.delta_norms = []          # ||theta_new - theta_old|| per iteration (Figure 3)
        self.converged = False

    @staticmethod
    def _add_bias(X):
        return np.hstack([X, np.ones((X.shape[0], 1))])   # feature 7: bias column of 1s

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = self._add_bias(np.asarray(X, float))
        y = np.asarray(y, float)
        m, n = X.shape
        if self.class_weight == "balanced":
            n_pos = max(y.sum(), 1.0)
            n_neg = max(m - y.sum(), 1.0)
            w = np.where(y == 1, m / (2 * n_pos), m / (2 * n_neg))
        else:
            w = np.ones(m)
        W = w.sum()
        R = self.l2 * np.eye(n)
        if not self.penalize_bias:
            R[-1, -1] = 0.0                       # the bias is the last column
        self.theta = np.zeros(n)
        self.delta_norms = []
        for _ in range(self.max_iter):
            p = self._sigmoid(X @ self.theta)
            grad = X.T @ (w * (p - y)) / W + R @ self.theta
            H = (X.T * (w * p * (1 - p))) @ X / W + R
            step = np.linalg.solve(H, grad)
            self.theta -= step
            norm = float(np.linalg.norm(step))
            self.delta_norms.append(norm)
            if norm < self.tol:
                self.converged = True
                break
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._add_bias(np.asarray(X, float)) @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def build_nn(batch_size=16, epochs=30, seed=42):
    """3 x 256 ReLU layers + sigmoid output (logistic output for binary MLP)."""
    return MLPClassifier(hidden_layer_sizes=(256, 256, 256), activation="relu",
                         solver="adam", learning_rate_init=0.001,
                         batch_size=batch_size, max_iter=epochs,
                         random_state=seed)


# ---------------------------------------------------------------------------
# Models for the daily walk-forward evaluation (src/walk_forward.py)
# ---------------------------------------------------------------------------
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import Ridge

CLASSIFIERS = ["majority", "logreg", "hgb", "rf", "mlp"]
REGRESSORS = ["mean", "ridge", "hgb"]


def make_classifier(name, seed=42):
    """Small, regularised models: a daily panel has only ~100 training rows."""
    if name == "majority":     # always predicts the training prior (the baseline to beat)
        return DummyClassifier(strategy="prior")
    if name == "logreg":       # our own Newton's-method logistic regression, class-balanced
        return NewtonLogisticRegression(max_iter=50, l2=0.1, class_weight="balanced",
                                        penalize_bias=False)
    if name == "hgb":          # gradient-boosted trees, shallow and heavily regularised
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=5,
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    if name == "mlp":          # tiny network + weight decay (the paper's 3x256 net would memorise 100 rows)
        return MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=300, random_state=seed)
    raise ValueError(name)


def make_regressor(name, seed=42):
    if name == "mean":         # baseline: always predict the training average
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=100,
                                             min_samples_leaf=10, l2_regularization=1.0,
                                             random_state=seed)
    raise ValueError(name)


def positive_proba(model, X):
    """P(class = 1) as a 1-D array for any of the models above."""
    p = model.predict_proba(X)
    p = np.asarray(p)
    if p.ndim == 2:
        return p[:, 1] if p.shape[1] > 1 else np.full(len(X), float(model.classes_[0]))
    return p
