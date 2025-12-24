import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional
import scipy as sp


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        num_objects, num_features = X.shape
        
        if self.loss_function is LossFunction.MSE:
            gradient = -2 * (X.T @ (y - X @ self.w) - self.l2_coef * self.w) / num_objects
        # elif self.loss_function is ...
        return gradient

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            num_objects, num_features = X.shape
            rmse = np.linalg.norm(X @ self.w - y) / num_objects
            return rmse ** 2
        # elif self.loss_function is ...
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        num_objects, num_features = X.shape
        self.w = np.zeros(num_features)
        
        if isinstance(self.optimizer, BaseDescent):
            self.loss_history.append(self.compute_loss(X, y))
            for _ in range(self.max_iter):
                diff = self.optimizer.step()
                self.loss_history.append(self.compute_loss(X, y))
                # if np.isnan(diff).sum(): 
                #     break
                # if  np.linalg.norm(diff) ** 2 > self.tolerance: 
                #     break

        elif self.optimizer is None:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y

        elif self.optimizer == 'SVD':
            u_matrix, sigma_matrix, vt_matrix = sp.sparse.linalg.svds(X, 4)
            self.w = (vt_matrix.T) @ ((u_matrix.T @ y) / sigma_matrix)














            
