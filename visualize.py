"""
visualize.py
Визуализация тестовых данных MNIST и предсказаний нейросети.
"""

import numpy as np
from PIL import Image
import mnist_loader
import network


def save_digit_image(pixel_vector, filename):
    """
    Сохраняет вектор пикселей как PNG-изображение.
    
    Параметры:
        pixel_vector — вектор 784x1 (значения от 0 до 1)
        filename — имя файла для сохранения
    """
    # Преобразуем вектор 784x1 обратно в матрицу 28x28
    pixels = pixel_vector.reshape(28, 28)
    
    # Преобразуем значения 0-1 в 0-255 для изображения
    pixels = (pixels * 255).astype(np.uint8)
    
    # Создаём изображение и сохраняем
    img = Image.fromarray(pixels, mode='L')
    
    # Увеличиваем для лучшей видимости (28x28 слишком маленькое)
    img = img.resize((280, 280), Image.NEAREST)
    
    img.save(filename)
    print(f"Изображение сохранено: {filename}")


def main():
    # Загружаем данные
    print("Загрузка данных...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Создаём и обучаем сеть
    print("Создание и обучение сети...")
    net = network.Network([784, 30, 10])
    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
    print()
    
    # Визуализируем несколько примеров из тестовой выборки
    print("=" * 50)
    print("Визуализация тестовых примеров:")
    print("=" * 50)
    
    for i in range(10):
        x, y = test_data[i]
        
        # Предсказание сети
        output = net.feedforward(x)
        predicted = np.argmax(output)
        confidence = output[predicted][0]
        
        # Сохраняем изображение
        filename = f"digit_{i}_label_{y}_predicted_{predicted}.png"
        save_digit_image(x, filename)
        
        # Выводим результат
        status = "✓" if predicted == y else "✗"
        print(f"  {status} Правильный ответ: {y}, Сеть предсказала: {predicted} (уверенность: {confidence:.2%})")
    
    print()
    print("Готово! PNG-файлы сохранены в текущей папке.")


if __name__ == "__main__":
    main()

