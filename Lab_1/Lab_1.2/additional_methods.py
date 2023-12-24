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

# промежуточная печать подсчётов методов
def write_temp_calculations(iteration, P, Q, temp_file):
    temp_file.write(f"\nИтерация {iteration}:\n")
    temp_file.write(f"P: {P:.5f}, Q: {Q:.5f} \n")