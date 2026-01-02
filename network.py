"""
network.py
Модуль создания и обучения нейронной сети для распознавания рукописных цифр
с использованием метода градиентного спуска.
"""

# Стандартные библиотеки
import random

# Сторонние библиотеки
import numpy as np


def sigmoid(z):
    """Сигмоидальная функция активации."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Производная сигмоидальной функции."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    """Класс для описания нейронной сети."""
    
    def __init__(self, sizes):
        """
        Конструктор класса.
        
        Параметры:
            sizes — список размеров слоев нейронной сети.
                    Например, [784, 30, 10] означает:
                    - входной слой: 784 нейрона
                    - скрытый слой: 30 нейронов
                    - выходной слой: 10 нейронов
        """
        self.num_layers = len(sizes)  # количество слоев нейронной сети
        self.sizes = sizes  # список размеров слоев нейронной сети
        
        # Случайные начальные смещения для каждого нейрона (кроме входного слоя)
        # sizes[1:] — это все слои кроме первого (входного)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        # Случайные начальные веса связей между слоями
        # zip(sizes[:-1], sizes[1:]) создает пары (размер предыдущего слоя, размер следующего слоя)
        # Матрица весов имеет размер (следующий слой × предыдущий слой)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """
        Прямое распространение сигнала через сеть.
        
        Параметры:
            a — входной вектор (например, 784x1 для изображения)
        
        Возвращает:
            Выходной вектор сети (например, 10x1 — вероятности для каждой цифры)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Обучение нейронной сети методом стохастического градиентного спуска.
        
        Параметры:
            training_data — список пар (x, y) для обучения
            epochs — количество эпох обучения
            mini_batch_size — размер подвыборки (мини-батча)
            eta — скорость обучения (learning rate)
            test_data — тестовые данные для оценки прогресса (опционально)
        """
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            # Перемешиваем обучающие данные
            random.shuffle(training_data)
            
            # Разбиваем на мини-батчи
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Обновляем веса и смещения для каждого мини-батча
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Выводим прогресс обучения
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        """
        Обновление весов и смещений сети на основе одного мини-батча.
        
        Параметры:
            mini_batch — список пар (x, y)
            eta — скорость обучения
        """
        # Инициализируем градиенты нулями
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Суммируем градиенты по всем примерам в мини-батче
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Обновляем веса и смещения
        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]
    
    def backprop(self, x, y):
        """
        Алгоритм обратного распространения ошибки.
        
        Параметры:
            x — входной вектор
            y — ожидаемый выходной вектор
        
        Возвращает:
            Кортеж (nabla_b, nabla_w) — градиенты для смещений и весов
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Прямой проход (feedforward) с сохранением промежуточных значений
        activation = x
        activations = [x]  # список активаций для каждого слоя
        zs = []  # список взвешенных сумм для каждого слоя
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Обратный проход (backpropagation)
        # Вычисляем ошибку выходного слоя
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Распространяем ошибку на скрытые слои
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        """
        Производная функции стоимости (квадратичной ошибки).
        
        Параметры:
            output_activations — выход сети
            y — ожидаемый выход
        
        Возвращает:
            Вектор частных производных
        """
        return (output_activations - y)
    
    def evaluate(self, test_data):
        """
        Оценка качества сети на тестовых данных.
        
        Параметры:
            test_data — список пар (x, y), где y — целое число (правильная цифра)
        
        Возвращает:
            Количество правильно распознанных примеров
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# Тело программы — выполняется при запуске файла напрямую
if __name__ == "__main__":
    import mnist_loader
    
    # Загружаем данные
    print("Загрузка данных MNIST...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"Загружено: {len(training_data)} обучающих, {len(test_data)} тестовых примеров")
    print()
    
    # Создаем нейронную сеть для распознавания цифр
    net = Network([784, 30, 10])
    print("Нейронная сеть создана: 784 -> 30 -> 10")
    print()
    
    # Запускаем обучение
    print("Начинаем обучение (30 эпох, mini-batch=10, eta=3.0)...")
    print()
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

