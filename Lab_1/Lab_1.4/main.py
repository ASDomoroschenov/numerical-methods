import numpy as np
from additional_methods import *

np.set_printoptions(precision = 4, suppress = True)

def jacobi_rotation(A, epsilon = 1e-6, max_iterations = 1000):
    if not is_symmetric(A):
        raise ValueError("Матрица A не является симметричной.")

    print_file = open("test/resources/temp_calculations.txt", "w")
    n = A.shape[0]
    V = np.identity(n)
    iteration = 0

    for iteration in range(max_iterations):
        max_val = 0.0
        sum_non_diagonal_elements = 0.0
        p, q = 0, 0

        for i in range(n):
            for j in range(i + 1, n):
                sum_non_diagonal_elements += A[i][j] ** 2
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        stop_iteration = np.sqrt(sum_non_diagonal_elements)

        if stop_iteration < epsilon:
            break

        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

        U = np.identity(n)
        U[p, p] = np.cos(theta)
        U[q, q] = np.cos(theta)
        U[p, q] = -np.sin(theta)
        U[q, p] = np.sin(theta)
        A = np.dot(np.dot(transpose(U), A), U)
        V = np.dot(V, U)

        write_temp_calculations(iteration + 1, A, V, U, max_val, p, q, theta, print_file)

    own_values = np.diag(A)
    own_vectors = V

    return own_values, own_vectors, iteration


if __name__ == '__main__':
    count_iteration = 0
    epsilon, matrix = read_matrix_and_epsilon("test/resources/test2.txt")
    matrix = np.array(matrix)
    own_values, own_vectors, count_iteration = jacobi_rotation(matrix, epsilon)

    with open('test/resources/result.txt', 'w') as file:
        sys.stdout = file
        print("Собственные значения:")
        print(own_values)
        print("Собственные векторы:")
        print(own_vectors)
        print("C точностью", epsilon)
        print("За", count_iteration, "итераций")

    sys.stdout = sys.__stdout__
    print("Собственные значения:")
    print(own_values)
    print("Собственные векторы:")
    print(own_vectors)
    print("C точностью", epsilon)
    print("За", count_iteration, "итераций\n")