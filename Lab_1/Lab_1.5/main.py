from additional_methods import *
np.set_printoptions(precision = 5, suppress = True)

def find_eigenvalues(A, epsilon = 1e-3):
    n = A.shape[0]
    own_values = []
    continue_qr = False
    i = 0
    while i < n - 1:
        if np.abs(A[i, i]) < epsilon:
            own_values.append(A[i, i])
            i += 1

        elif np.abs(A[i+1][i]) < epsilon:
            own_values.append(A[i][i])
            i += 1

        else:
            minor = A[i:i+2, i:i+2]
            a = 1.0
            b = -minor[0, 0] - minor[1, 1]
            c = minor[0, 0] * minor[1, 1] - minor[0, 1] * minor[1, 0]
            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                continue_qr = True
                return np.array(own_values), continue_qr
            else:
                real_part = -b / (2 * a)
                imaginary_part = np.sqrt(-discriminant) / (2 * a)
                complex_root1 = complex(real_part, imaginary_part)
                complex_root2 = complex(real_part, -imaginary_part)
                own_values.append(complex_root1)
                own_values.append(complex_root2)

            i += 2

    if i == n - 1:
        own_values.append(A[i][i])

    return np.array(own_values), continue_qr

def qr_algorithm(A, epsilon = 1e-3, max_iterations = 1000):
    n = A.shape[0]
    own_values = np.zeros(n)
    iteration = 0
    print_file = open("test/resources/temp_calculations.txt", "w")

    for iteration in range(max_iterations):
        Q, R, H = qr_decomposition(A)
        A = dot(R, Q)
        write_temp_calculations(iteration + 1, R, Q, H, A, print_file)
        if np.all(np.abs(A - np.triu(A, -1)) < epsilon):
            break

    continue_iteration = True
    while iteration < max_iterations and continue_iteration == True:
        Q, R, H = qr_decomposition(A)
        A = dot(R, Q)
        write_temp_calculations(iteration + 1, R, Q, H, A, print_file)
        own_values, continue_iteration = find_eigenvalues(A, epsilon)

    prev_own_values = own_values
    continue_iteration = True

    while continue_iteration == True:
        Q, R, H = qr_decomposition(A)
        A = dot(R, Q)
        write_temp_calculations(iteration + 1, R, Q, H, A, print_file)
        new_own_values, continue_iteration = find_eigenvalues(A, epsilon)
        eigenvalue_difference = np.abs(new_own_values - prev_own_values)
        if np.all(eigenvalue_difference < epsilon):
            continue_iteration = False

    own_values = new_own_values

    return own_values, R, Q, A, iteration + 1

def qr_decomposition(A):
    n = A.shape[0]
    Q = np.identity(n)
    R = A.copy()

    for i in range(n - 1):
        x = R[i:, i]
        v = np.zeros(n)
        v[i] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
        v[i + 1:] = x[-len(x) + 1:]
        H = np.identity(n)
        H -= 2 * np.outer(v, v) / dot(transpose(v), v)
        R = dot(H, R)
        Q = dot(Q, H)

    return Q, R, H


if __name__ == '__main__':
    epsilon, matrix = read_matrix_and_epsilon("test/resources/test1.txt")
    own_values, R, Q, A, count_iteration = qr_algorithm(matrix, epsilon)

    with open('test/resources/result.txt', 'w') as file:
        sys.stdout = file
        print("Собственные значения:")
        print(own_values)
        print("Матрица R:")
        print(R)
        print("Матрица Q:")
        print(Q)
        print("Матрица A:")
        print(A)
        print("C точностью", epsilon)
        print("За", count_iteration, "итераций")

    sys.stdout = sys.__stdout__
    print("Собственные значения:")
    print(own_values)
    print("Матрица R:")
    print(R)
    print("Матрица Q:")
    print(Q)
    print("Матрица A:")
    print(A)
    print("C точностью", epsilon)
    print("За", count_iteration, "итераций")