import numpy as np
from additional_methods import *

np.set_printoptions(precision=4, suppress=True)

def run_through_method(A, b):
    print_file = open("test/resources/temp_calculations.txt", "w")
    n = len(A)
    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = -A[0, 1] / A[0, 0]
    Q[0] = b[0] / A[0, 0]

    for i in range(1, n - 1):
        denominator = A[i, i] + A[i, i - 1] * P[i - 1]
        P[i] = -A[i, i + 1] / denominator
        Q[i] = (b[i] - A[i, i - 1] * Q[i - 1]) / denominator
        write_temp_calculations(i, P[i], Q[i], print_file)

    x = np.zeros(n)
    denominator = A[n - 1, n - 1] + A[n - 1, n - 2] * P[n - 2]
    Q[n - 1] = (b[n - 1] - A[n - 1, n - 2] * Q[n - 2]) / denominator
    x[n - 1] = Q[n - 1]

    write_temp_calculations(i + 1, P[i + 1], Q[i + 1], print_file)

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


if __name__ == '__main__':
    matrix, b_column = read_matrix_and_vector("test/resources/test1.txt")
    result = run_through_method(matrix, b_column)

    with open('test/resources/result.txt', 'w') as file:
        sys.stdout = file
        print(result)

    sys.stdout = sys.__stdout__
    print(result)
    print(np.matmul(matrix, result) - b_column)
