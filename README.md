# Python-Neural-Network-in-NumPY

## framework.py
Включает в себя слои:
* Линейный слой
* Функции активации:
1. Sigmoid
2. ReLU
3. LeakyRELU
4. Softmax
* Слои регуляризации:
1. Dropout
2. BatchNorm
* Функции потерь:
1. MSE
2. CrossEntropy

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
