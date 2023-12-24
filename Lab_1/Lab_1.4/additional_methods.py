import sys
import numpy as np

#Выводим результаты с точностью 4 знака после запятой
np.set_printoptions(precision = 4, suppress = True)

# Считывание матрицы A и epsilon из файла
def read_matrix_and_epsilon(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        epsilon = None

        for line in lines:
            line = line.strip()

            # Если строка не пустая
            if line:
                # Разделяем строку на элементы
                values = list(map(float, line.split()))

                # Если одно значение, это epsilon
                if len(values) == 1 and epsilon is None:
                    epsilon = float(line)
                # Если больше одного значения, это матрица A
                elif len(values) > 1:
                    matrix.append(values)

    return epsilon, np.array(matrix)

# промежуточная печать подсчётов методов
def write_temp_calculations(iteration, A, V, U, max_val, p, q, theta, temp_file):
    temp_file.write(f"Итерация {iteration}:\n")
    temp_file.write(f"Матрица A:\n{A}\n")
    temp_file.write(f"Матрица V (собственные векторы):\n{V}\n")
    temp_file.write(f" Матрица вращения U:\n {U}\n")
    temp_file.write(f"Макс недиагональный элемент (до обновления матрицы А): {max_val:.4f}\n")
    temp_file.write(f"Индексы p, q: ({p}, {q})\n")
    temp_file.write(f"Угол поворота (тетта): {theta:.4f}\n\n")

# Транспонирование матрицы или вектора
def transpose(matrix):
    # Если первый элемент в матрице список или массив numpy, это двумерная матрица
    if isinstance(matrix[0], list) or isinstance(matrix[0], np.ndarray):
        rows = len(matrix)
        cols = len(matrix[0])

        transposed_matrix = np.eye(cols, rows)

        for i in range(rows):
            for j in range(cols):
                transposed_matrix[j][i] = matrix[i][j]

    # Если первый элемент - не список, это одномерный вектор
    # (для столбца матрицы работает аналогично, хз почему так в numpy)
    else:
        rows = len(matrix)

        matrix_dtype = type(matrix[0]) if rows > 0 else None
        transposed_matrix = np.zeros(rows, dtype=matrix_dtype)

        for i in range(rows):
            transposed_matrix[i] = matrix[i]

    return transposed_matrix

# проверка на симметричность матрицы
def is_symmetric(matrix):
    return (matrix == transpose(matrix)).all()

# Функция для умножения двух массивов (урезанный аналог np.dot)
def dot(a, b):
    # .ndim - размерность массива, 1 - вектор, 2 - матрица
    # .shape возвращает [кол-во строк, кол-во столбцов] для матрицы
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