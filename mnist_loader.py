"""
mnist_loader.py
Модуль загрузки данных MNIST для обучения нейронной сети распознаванию рукописных цифр.
"""

# Стандартные библиотеки
import gzip
import pickle

# Сторонние библиотеки
import numpy as np


def load_data():
    """
    Загружает данные MNIST из сжатого файла mnist.pkl.gz.
    
    Возвращает кортеж (training_data, validation_data, test_data):
    - training_data: 50 000 изображений для обучения
    - validation_data: 10 000 изображений для валидации
    - test_data: 10 000 изображений для тестирования
    """
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    Преобразует цифру j (0-9) в 10-мерный вектор-столбец,
    где единица стоит в позиции j, а остальные элементы нулевые.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    """
    Загружает и преобразует данные MNIST в формат, пригодный для нейросети.
    
    Возвращает кортеж (training_data, validation_data, test_data):
    - training_data: список пар (x, y), где x — вектор 784x1 (изображение),
      y — вектор 10x1 (метка в формате one-hot)
    - validation_data: список пар (x, y), где x — вектор 784x1,
      y — целое число (метка 0-9)
    - test_data: список пар (x, y), где x — вектор 784x1,
      y — целое число (метка 0-9)
    """
    tr_d, va_d, te_d = load_data()
    
    # Преобразование обучающих данных
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Преобразование валидационных данных
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Преобразование тестовых данных
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

