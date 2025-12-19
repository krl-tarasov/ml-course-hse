import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        # TODO: реализовать формулу затухающего шага обучения
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        diff = self.update_weights()
        self.iteration += 1
        return diff


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        gradient = self.model.compute_gradients(X_train, y_train)

        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - lr * gradient
        return (-lr * gradient)


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели

        X_train = self.model.X_train
        y_train = self.model.y_train

        random_indices = np.random.choice(y_train.shape[0], self.batch_size, replace = False)
        X_batch, y_batch = X_train[random_indices], y_train[random_indices]
        gradient = self.model.compute_gradients(X_batch, y_batch)

        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - lr * gradient
        return (-lr * gradient)


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        
        if self.grad_memory is None:
            self.grad_sum = np.zeros(num_features, float)
            self.grad_memory = np.zeros([num_objects, num_features], float)

        current_object_index = [self.iteration % num_objects]

        X = X_train[current_object_index]
        y = y_train[current_object_index]

        new_current_grad = self.model.compute_gradients(X, y)

        old_current_grad = self.grad_memory[current_object_index].copy()
        self.grad_memory[current_object_index] = new_current_grad
        grad_diff = ((new_current_grad - old_current_grad) / num_objects).squeeze()
        self.grad_sum = self.grad_sum + grad_diff

        lr = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - lr * self.grad_sum

        return (- lr * self.grad_sum)

class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        _, num_features = X_train.shape
        gradient = self.model.compute_gradients(X_train, y_train)

        if self.velocity is None:
            self.velocity = np.zeros(num_features)

        lr = self.lr_schedule.get_lr(self.iteration)
        self.velocity = self.beta * self.velocity + lr * gradient

        self.model.w = self.model.w - self.velocity
        return -self.velocity

class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        _, num_features = X_train.shape
        gradient = self.model.compute_gradients(X_train, y_train)

        if self.m is None or self.v is None:
            self.m = np.zeros(num_features)
            self.v = np.zeros(num_features)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta1 * self.v + (1 - self.beta1) * gradient ** 2

        m_norm = self.m / (1 - self.beta1) ** self.iteration
        v_norm = self.v / (1 - self.beta2) ** self.iteration

        lr = self.lr_schedule.get_lr(self.iteration)
        diff = lr * (m_norm) / (np.sqrt(v_norm) + self.eps)

        self.model.w = self.model.w - diff

        return diff   