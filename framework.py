import numpy as np


class Module():
    def __init__(self):
        self._train = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError

    def parameters(self):
        'Возвращает список собственных параметров.'
        return []

    def grad_parameters(self):
        'Возвращает список тензоров-градиентов для своих параметров.'
        return []

    def train(self):
        self._train = True

    def eval(self):
        self._train = False


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        """
        Прогоните данные последовательно по всем слоям:

            y[0] = layers[0].forward(input)
            y[1] = layers[1].forward(y_0)
            ...
            output = module[n-1].forward(y[n-2])

        Это должен быть просто небольшой цикл: for layer in layers...

        Хранить выводы ещё раз не надо: они сохраняются внутри слоев после forward.
        """

        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        """
        Backward -- это как forward, только наоборот. (с)

        Предназначение backward:
        1. посчитать градиенты для собственных параметров
        2. передать градиент относительно своего входа

        О своих параметрах модули сами позаботятся. Нам же нужно позаботиться о передачи градиента.

            g[n-1] = layers[n-1].backward(y[n-2], grad_output)
            g[n-2] = layers[n-2].backward(y[n-3], g[n-1])
            ...
            g[1] = layers[1].backward(y[0], g[2])
            grad_input = layers[0].backward(input, g[1])

        Тут цикл будет уже чуть посложнее.
        """

        for i in range(len(self.layers) - 1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i - 1].output, grad_output)

        grad_input = self.layers[0].backward(input, grad_output)

        return grad_input

    def parameters(self):
        'Можно просто сконкатенировать все параметры в один список.'
        res = []
        for l in self.layers:
            res += l.parameters()
        return res

    def grad_parameters(self):
        'Можно просто сконкатенировать все градиенты в один список.'
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Xavier initialization: инциализируем так,
        # что если на вход идет N(0, 1)
        # то и на выходе должно идти N(0, 1)
        stdv = 1. / np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        self.b = np.random.uniform(-stdv, stdv, size=dim_out)
        self.velocity = [np.zeros(shape=self.W.shape), np.zeros(shape=self.b.shape)]
        self.buffer = [np.zeros(shape=self.W.shape), np.zeros(shape=self.b.shape)]

    def forward(self, input):
        self.output = np.dot(input, self.W) + self.b
        return self.output

    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)

        #     in_dim x batch_size
        self.grad_W = np.dot(input.T, grad_output)
        #                 batch_size x out_dim
        self.grad_W /= input.shape[0]
        grad_input = np.dot(grad_output, self.W.T)

        return grad_input

    def parameters(self):
        # return [self.W, self.b], self.velocity, self.buffer
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.grad_W, self.grad_b]


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        self.output = np.where(input <= 0, input, self.slope * input)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.choose(input <= 0, [np.sign(input) * self.slope, 1]) * grad_output
        return grad_input

    # def forward(self, input):
    #     self.output = np.maximum(input, np.multiply(self.slope, input))
    #     return self.output

    # def backward(self, input, grad_output):
    #     grad_input = np.multiply(grad_output, input > 0) + np.multiply(self.slope, np.multiply(grad_output, input < 0))
    #     return grad_input

# class Tanh(Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input):
#         self.output = np.tanh(input)
#         return self.output

#     def backward(self, input, grad_output):
#         grad_input = 1-np.tanh(input)**2
#         return grad_input


class Sigmoid(Module):
    def __init__(self, slope=0.03):
        super().__init__()

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(self.forward(input), 1 - self.forward(input))
        return grad_input


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p
        self.mask = None

    def forward(self, input):
        if self._train:
            self.mask = np.random.binomial(1, (1-self.p), size=input.shape)
            self.output = self.mask * input
        else:
            self.output = input * (1-self.p)
        return self.output

    def backward(self, input, grad_output):
        if self._train:
            # mask = np.random.binomial(1, self.p, size=input.shape)
            self.grad_input = self.mask * grad_output
        else:
            self.grad_input = grad_output
        return self.grad_input


class BatchNorm(Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.mu = np.zeros(shape=num_features)
        self.sigma = np.ones(shape=num_features)
        self.momentum = 0.9
        self.eps = 1e-5

    def forward(self, input):
        if self._train:
            mu_new = np.mean(input, axis=0)
            sigma_new = np.mean((input - mu_new) ** 2, axis=0)
            # sigma_new = np.var(input, axis=0, ddof=1)
            self.mu = self.momentum * self.mu + (1 - self.momentum) * mu_new
            self.sigma = self.momentum * self.sigma + (1 - self.momentum) * sigma_new
            input_norm = (input - mu_new) / np.sqrt(sigma_new + self.eps)
            self.output = self.gamma * input_norm + self.beta
        else:
            input_norm = (input - self.mu) / np.sqrt(self.sigma + self.eps)
            self.output = self.gamma * input_norm + self.beta
        return self.output

    def backward(self, input, grad_output):
        if self._train:
            # grad_output = grad_output.ravel().reshape(grad_output.shape[0], -1)
            mu = np.mean(input, axis=0)
            sigma = np.mean((input - mu) ** 2, axis=0)
            input_norm = (input - mu) / np.sqrt(sigma + self.eps)
            t = 1. / np.sqrt(sigma + self.eps)
            m = input.shape[0]
            self.grad_gamma = np.sum(grad_output * input_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)
            grad_x = (self.gamma * t / m) * (m * grad_output - t ** 2 * (input - mu) * np.sum(grad_output * (input - mu), axis=0) - np.sum(grad_output, axis=0))
            grad_input = grad_x

        else:
            # grad_output = grad_output.ravel().reshape(grad_output.shape[0], -1)

            input_norm = (input - self.mu) / np.sqrt(self.sigma + self.eps)
            t = 1. / np.sqrt(self.sigma + self.eps)
            m = input.shape[0]
            self.grad_gamma = np.sum(grad_output * input_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)
            grad_x = (self.gamma * t / m) * (m * grad_output - t ** 2 * (input - self.mu) * np.sum(grad_output * (input - self.mu), axis=0) - np.sum(grad_output, axis=0))
            grad_input = grad_x

        return grad_input



    def parameters(self):
        return [self.gamma, self.beta]

    def grad_parameters(self):
        return [self.grad_gamma, self.grad_beta]


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sub_input = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(sub_input) / np.sum(np.exp(sub_input), axis=1, keepdims=True)
        return self.output

    def backward(self, input, grad_output):
        return grad_output

class Criterion():
    def forward(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        raise NotImplementedError


class MSE(Criterion):
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output

    def backward(self, input, target):
        self.grad_output = (input - target) * 2 / input.shape[0]
        return self.grad_output


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)
        self.output = -1 * np.sum(np.multiply(np.log(SoftMax().forward(input_clamp)), target)) / input_clamp.shape[0]
        return self.output

    def backward(self, input, target):
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)

        grad_output = (SoftMax().forward(input_clamp) - target)

        # Чтобы градиент сходился с градиентом торча, раскомментить:
        # grad_input = (SoftMax().forward(input_clamp) - target) / input_clamp.shape[0]

        return grad_output
