import sys
import numpy as np

# Выводим результаты с точностью 2 знака после запятой
np.set_printoptions(precision = 2, suppress = True)

# Считывание матрицы A и столбца свободных членов b из файла
def read_matrix_and_vector(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        vector = []
        epsilon = None

        for line in lines:
            line = line.strip()

            # Если строка не пустая
            if line:
                # Если это число (точность epsilon)
                if epsilon is None:
                    epsilon = float(line)
                else:
                    # Разделяем строку на элементы
                    values = list(map(float, line.split()))

                    # Проверяем, сколько значений в строке
                    if len(values) > 1:
                        # Если больше одного значения, это матрица A
                        matrix.append(values)
                    elif len(values) == 1:
                        # Если одно значение, это столбец b
                        vector.append(values[0])

    return epsilon, np.array(matrix), np.array(vector)

# метод проверки матрицы на сходимость по методу Якоби и Зейделя
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        sum_of_other_elements = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= sum_of_other_elements:
            return False
    return True

# промежуточная печать подсчётов методов
def write_temp_calculations(iteration, x, temp_file):
    temp_file.write(f"\nИтерация {iteration}:\n")
    for i, value in enumerate(x):
        temp_file.write(f"x[{i}] = {value:.4f}\n")