import numpy as np

# функция генреации батча
def loader(X, Y, batch_size):
    n = X.shape[0]

    # в начале каждой эпохи будем всё перемешивать
    # важно, что мы пермешиваем индексы, а не X
    indices = np.arange(n)
    np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        # в конце нам, возможно, нужно взять неполный батч
        end = min(start + batch_size, n)

        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]

class SGD():
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, params, gradients):
        for weights, gradient in zip(params, gradients):
            # if weights.shape != gradient.shape:
            #     gradient = gradient[:, np.newaxis]
            weights -= self.lr * gradient

class SGD_momentum():
    def __init__(self, params, lr=1e-3, momentum=0.9):
        # self.params = params
        # self.buffer = [np.zeros(shape=elem.shape) for elem in params]
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros(shape=elem.shape) for elem in params]

    def step(self, parameters, gradients):
        # if type(params[0]) is list:
        #     params_lin, velocity, _ = params
        # else:
        #     params_lin = params
        # # momentum = 0.0
        for weights, vel, gradient in zip(parameters, self.velocity, gradients):
            vel = self.momentum * vel + gradient
            weights -= self.lr * vel


class RMSProp():
    def __init__(self, params, lr=1e-3, momentum=0.0, alpha=0.75, eps=1e-08):
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.buffer = [np.zeros(shape=elem.shape) for elem in params]
        self.velocity = [np.zeros(shape=elem.shape) for elem in params]

    def step(self, parameters, gradients):
        for weights, vel, buf, gradient in zip(parameters, self.velocity, self.buffer, gradients):
            vel = self.alpha * vel + (1 - self.alpha) * gradient**2
            buf = self.momentum * buf + gradient / (np.sqrt(vel) + self.eps)
            weights -= self.lr * buf
            # buf = alpha * buf + (1 - alpha) * gradient**2
            # weights -= lr * gradient / (np.sqrt(buf+eps))

class Adam():
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-08):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.buffer = [np.zeros(shape=elem.shape) for elem in params]
        self.velocity = [np.zeros(shape=elem.shape) for elem in params]

    def step(self, parameters, gradients):

        for weights, vel, buf, gradient in zip(parameters, self.velocity, self.buffer, gradients):
            vel = self.beta1 * vel + (1 - self.beta1) * gradient
            buf = self.beta2 * buf + (1 - self.beta2) * gradient ** 2
            vel_t = vel / (1 - self.beta1)
            buf_t = buf / (1 - self.beta2)
            weights -= self.lr * vel_t / (np.sqrt(buf_t) + self.eps)


# def RMSProp(params, gradients, lr=1e-3, momentum=0.0, alpha=0.75, eps=1e-08):
#     if type(params[0]) is list:
#         params_lin, velocity, buffer = params
#     else:
#         params_lin = params
#     # momentum = 0.0
#     for weights, vel, buf, gradient in zip(params_lin, velocity, buffer, gradients):
#
#         vel = alpha * vel + (1 - alpha) * gradient**2
#         buf = momentum * buf + gradient / (np.sqrt(vel) + eps)
#         weights -= lr * buf
#         # buf = alpha * buf + (1 - alpha) * gradient**2
#         # weights -= lr * gradient / (np.sqrt(buf+eps))

# def Adam(params, gradients, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-08):
#     if type(params[0]) is list:
#         params_lin, velocity, buffer = params
#     else:
#         params_lin = params
#     # momentum = 0.0
#     for weights, vel, buf, gradient in zip(params_lin, velocity, buffer, gradients):
#
#         vel = beta1 * vel + (1 - beta1) * gradient
#         buf = beta2 * buf + (1 - beta2) * gradient**2
#         vel_t = vel / (1 - beta1)
#         buf_t = buf / (1 - beta2)
#         weights -= lr * vel_t / (np.sqrt(buf_t)+eps)

# def SGD_momentum(self, params, gradients, lr=1e-3, momentum=0.9):
#     if type(params[0]) is list:
#         params_lin, velocity, _ = params
#     else:
#         params_lin = params
#     # momentum = 0.0
#     for weights, vel, gradient in zip(params, self.velocity, gradients):
#         vel = momentum * vel + gradient
#         weights -= lr * vel

# def SGD(self, params, gradients, lr=1e-3):
#     if type(params[0]) is list:
#         params_lin, _, _ = params
#     else:
#         params_lin = params
#     for weights, gradient in zip(params_lin, gradients):
#         #print(type(lr), type(gradient))
#         #print(lr, gradient)
#         weights -= lr * gradient