# Python-Neural-Network-in-NumPY

## framework.py
Включает в себя слои:
* Linear
* Функции активации:
** Sigmoid
** ReLU
** LeakyRELU
** Softmax
* Слои регуляризации
** Dropout
** BatchNorm
* Функции потерь
** MSE
** CrossEntropy

## optim.py
Включает в себя оптимизаторы:
* SGD
* SGD with Nesterov momentum
* RMSProp
* Adam
 а также генератор батча:
 * loader

## Example_MNIST.ipynb

Демонстрация работы, написанного фреймворка. Включает в себя два простых примера на сгенерированных данных: линейная и логистическая регрессия, а также классификацию датасета MNIST (точность модели 98%).
