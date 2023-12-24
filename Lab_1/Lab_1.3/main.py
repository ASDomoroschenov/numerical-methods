from additional_methods import *

np.set_printoptions(precision = 2, suppress = True)

def jacobi_iteration(A, b, eps = 1e-3, max_iterations = 1000):
    print_file_jacobi = open("test/resources/temp_calculations_jacobi.txt", "w")
    n = len(A)
    iteration_count_Jacobi = 0
    x = np.zeros(n)

    if not is_diagonally_dominant(A):
        raise Exception("Матрица A не является диагонально доминирующей, метод Якоби может не сойтись.")

    for iteration in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if i != j:
                    sum_term += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_term) / A[i, i]
        if np.linalg.norm(x_new - x) < eps:
            return x_new, iteration_count_Jacobi
        x = x_new
        iteration_count_Jacobi += 1
        write_temp_calculations(iteration_count_Jacobi, x, print_file_jacobi)

    raise Exception("Метод Якоби не сошелся после", iteration_count_Jacobi, "числа итераций")

def seidel_iteration(A, b, eps = 1e-3, max_iterations = 1000):
    print_file_seidel = open("test/resources/temp_calculations_seidel.txt", "w")
    n = len(A)
    x = np.zeros(n)
    iteration_count_Seidel = 0

    if not is_diagonally_dominant(A):
        raise Exception("Матрица A не является диагонально доминирующей, метод Зейделя может не сойтись.")

    for iteration in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            sum_term = 0
            for j in range(i):
                sum_term += A[i, j] * x_new[j]
            for j in range(i + 1, n):
                sum_term += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_term) / A[i, i]
        if np.linalg.norm(x_new - x) < eps:
            return x_new, iteration_count_Seidel
        x = x_new
        iteration_count_Seidel += 1
        write_temp_calculations(iteration_count_Seidel, x, print_file_seidel)

    raise Exception("Метод Зейделя не сошелся после", iteration_count_Seidel, "числа итераций")


if __name__ == '__main__':
    epsilon, matrix, b_column = read_matrix_and_vector("test/resources/test3.txt")
    result_Jacobi, iteration_count_Jacobi = jacobi_iteration(matrix, b_column, epsilon)
    result_Seidel, iteration_count_Seidel = seidel_iteration(matrix, b_column, epsilon)

    with open('test/resources/result.txt', 'w') as file:
        sys.stdout = file
        print("Решение методом Якоби:")
        print(result_Jacobi, "с точностью", epsilon)
        print("За", iteration_count_Jacobi, "итераций")
        print("\nРешение методом Зейделя:")
        print(result_Seidel, "с точностью", epsilon)
        print("За", iteration_count_Seidel, "итераций")

    sys.stdout = sys.__stdout__
    print("Решение методом Якоби:")
    print(result_Jacobi, "с точностью", epsilon)
    print("За", iteration_count_Jacobi, "итераций")
    print("\nРешение методом Зейделя:")
    print(result_Seidel, "с точностью", epsilon)
    print("За", iteration_count_Seidel, "итераций")