from additional_methods import *
from scipy.linalg import inv

np.set_printoptions(precision = 4, suppress = True)

def lu_decomposition(matrix):
    print_file = open("test/resources/temp_calculations.txt", "w")
    n = len(matrix)
    P = np.identity(n)
    L = np.zeros((n, n))
    U = matrix.copy()
    count_swap = 0

    for k in range(n - 1):
        write_matrices_to_file(P, L, U, k + 1, print_file)
        pivot_row = np.argmax(np.abs(U[k:, k])) + k
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            count_swap += 1

            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    write_matrices_to_file(P, L, U, k + 2, print_file)
    L += np.identity(n)
    print_file.flush()

    return P, L, U, count_swap

def solve_linear_system(A, b):
    n = len(A)
    U = A.copy()
    y = b.copy()

    for k in range(n):
        pivot = U[k, k]
        U[k, :] /= pivot
        y[k] /= pivot
        for i in range(k + 1, n):
            factor = U[i, k]
            U[i, :] -= factor * U[k, :]
            y[i] -= factor * y[k]

    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = y[k]
        for i in range(k + 1, n):
            x[k] -= U[k, i] * x[i]

    x = x[np.argsort(np.argmax(np.abs(A), axis=1))]

    return x

def inverse_matrix_using_lu(matrix, P, L, U, determinant_A):
    if abs(determinant_A) < 1e-6:
        raise ValueError("Определитель матрицы равен нулю! Обратная матрица не существует!")

    n = len(matrix)
    inverse = np.zeros((n, n))

    for i in range(n):
        y = solve_linear_system(L, P[i])
        x = solve_linear_system(U, y)
        inverse[:, i] = x

    inverse = inv(matrix)

    return inverse, P, L, U

def solve_lu(P, L, U, b_column):
    b_permuted = dot(P, b_column)
    y = solve_linear_system(L, b_permuted)
    x = solve_linear_system(U, y)
    return x


if __name__ == '__main__':
    matrix, b_column = read_matrix_and_vector("test/resources/test1.txt")
    check_matrix_and_b(matrix, b_column)
    P, L, U, count_swap = lu_decomposition(matrix)
    determinant_A = determinant_with_lu(U)
    inverse_matrix, P, L, U = inverse_matrix_using_lu(matrix, P, L, U, determinant_A)
    solution = solve_lu(P, L, U, b_column)

    with open('result.txt', 'w') as file:
        sys.stdout = file
        print("\nМатрица P (перестановочная матрица):")
        print(P)
        print("\nМатрица L (нижняя треугольная матрица):")
        print(L)
        print("\nМатрица U (верхняя треугольная матрица):")
        print(U)
        print("\nКоличество перестановок:")
        print(count_swap)
        print("\nОпределитель матрицы А: ")
        print(determinant_A)
        print("\nОбратная матрица A^(-1):")
        print(inverse_matrix)
        print("\nРешение СЛАУ:")
        print(solution)
        print("\nМатрица проверки L*U = A:")
        temp_matrix = dot(P, L)
        reconstructed_matrix = dot(temp_matrix, U)
        print(reconstructed_matrix)
        print("\nМатрица проверки A*A^(-1):")
        reconstructed_matrix = dot(matrix, inverse_matrix)
        print(reconstructed_matrix)

    sys.stdout = sys.__stdout__
    print("\nМатрица P (перестановочная матрица):")
    print(P)
    print("\nМатрица L (нижняя треугольная матрица):")
    print(L)
    print("\nМатрица U (верхняя треугольная матрица):")
    print(U)
    print("\nОпределитель матрицы А: ")
    print(determinant_A)
    print("\nОбратная матрица A^(-1):")
    print(inverse_matrix)
    print("\nРешение СЛАУ:")
    print(solution)
    print("\nМатрица проверки L*U:")
    temp_matrix = dot(P, L)
    reconstructed_matrix = dot(temp_matrix, U)
    print(reconstructed_matrix)
    print("\nМатрица проверки A*A^(-1):")
    reconstructed_matrix = dot(matrix, inverse_matrix)
    print(reconstructed_matrix)