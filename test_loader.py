import mnist_loader

# Загружаем данные
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Проверяем размеры наборов данных
print("Размер обучающей выборки:", len(training_data))
print("Размер валидационной выборки:", len(validation_data))
print("Размер тестовой выборки:", len(test_data))

# Проверяем формат первого элемента обучающей выборки
x, y = training_data[0]
print("Размер входного вектора x:", x.shape)
print("Размер выходного вектора y:", y.shape)

# Проверяем формат первого элемента тестовой выборки
x_test, y_test = test_data[0]
print("Размер тестового x:", x_test.shape)
print("Метка тестового примера:", y_test)