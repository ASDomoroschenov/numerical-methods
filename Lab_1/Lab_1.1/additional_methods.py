import sys
import numpy as np

# Выводим результаты с точностью 4 знака после запятой
np.set_printoptions(precision = 4, suppress = True)

# Считывание матрицы A и столбца свободных членов b из файла
def read_matrix_and_vector(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        vector = []

        for line in lines:
            # Разделяем строку на элементы
            values = list(map(float, line.strip().split()))

            # Проверяем, сколько значений в строке
            if len(values) > 1:
                # Если больше одного значения, это матрица A
                matrix.append(values)
            elif len(values) == 1:
                # Если одно значение, это столбец b
                vector.append(values[0])

        return np.array(matrix), np.array(vector)

# метод проверки матрицы A и столбца b
def check_matrix_and_b(matrix, b_column):
    num_rows = len(matrix)
    num_columns = len(matrix[0]) if num_rows > 0 else 0

    if num_rows != num_columns:
        raise ValueError("Матрица не является квадратной!")

    if num_rows != len(b_column):
        raise ValueError("Число строк в матрице не соответствует числу элементов столбца b!")

# Функция для нахождения определителя матрицы
def determinant_with_lu(U):
    det_U = round(np.prod(np.diagonal(U)), 7)
    return det_U

# Функция для промежуточной записи матриц P, L и U в файл
def write_matrices_to_file(P, L, U, step, temp_file):
    temp_file.write(f"Шаг {step}:")
    temp_file.write(f"\nМатрица P:\n{P}")
    temp_file.write(f"\nМатрица L:\n{L}")
    temp_file.write(f"\nМатрица U:\n{U}\n\n")

# Функция для умножения двух массивов (урезанный аналог np.dot)
def dot(a, b):
    # ndim - размерность массива, 1 - вектор, 2 - матрица
    # shape возвращает [кол-во строк, кол-во столбцов] для матрицы
    # Всего 4 случая:

    # 1. Умножение двух одномерных векторов
    if a.ndim == 1 and b.ndim == 1:
        if len(a) != len(b):
            raise ValueError("Векторы должны быть одинаковой длины")

        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]

    # 2. Умножение двух матриц
    elif a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError("Размеры матриц не совпадают")

        result = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    result[i, j] += a[i, k] * b[k, j]

    # 3. Умножение одномерного вектора на матрицу
    elif a.ndim == 1 and b.ndim == 2:
        if len(a) != b.shape[0]:
            raise ValueError("Длина вектора должна равняться количеству строк матрицы")

        result = np.zeros(b.shape[1])
        for i in range(b.shape[1]):
            for j in range(len(a)):
                result[i] += a[j] * b[j, i]

    # 4. Умножение матрицы на одномерный вектор
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != len(b):
            raise ValueError("Количество столбцов матрицы должно совпадать с длиной вектора")

        result = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                result[i] += a[i, j] * b[j]

    else:
        raise ValueError("Такие данные пока не обрабатываются")

    return result