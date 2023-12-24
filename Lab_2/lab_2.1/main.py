import math
import sys
import numpy as np
from additional_methods import *

np.set_printoptions(precision = 5, suppress = True)

def simple_iteration_solver(f, x0, epsilon, max_iterations = 100):
    x = x0
    iteration = 0

    while iteration < max_iterations:
        next_x = f(x)
        if abs(next_x - x) < epsilon:
            return next_x, iteration
        x = next_x
        iteration += 1

    raise Exception("Метод не сошелся после максимального числа итераций.")

def newton_method(f, df, x0, epsilon = 1e-6, max_iterations = 100):
    x = x0
    iteration = 0

    while iteration < max_iterations:
        delta_x = f(x) / df(x)
        x = x - delta_x
        if abs(delta_x) < epsilon:
            return x, iteration
        iteration += 1

    raise ValueError("Метод Ньютона не сошелся. Увеличьте количество итераций или проверьте начальное приближение.")

def f(x):
    return math.pow(x, 3) - 2 * math.pow(x, 2) - 10 * x + 15

def phi(x):
    return (math.pow(x, 3) - 2 * math.pow(x, 2) + 15) / 10

def df(x):
    return 3 * math.pow(x, 2) - 4 * x - 10


if __name__ == '__main__':
    x0 = 0.5
    epsilon = 0.001

    result_iteration, count_iteration = simple_iteration_solver(phi, x0, epsilon)
    result_newton, count_newton = newton_method(f, df, x0, epsilon)

    with open('result.txt', 'w') as file:
        sys.stdout = file
        print("По методу простых итреаций:")
        print("корень =", result_iteration, "за", count_iteration, "итераций", "с точностью", epsilon)
        print("\nПо методу Ньютона:")
        print("корень =", result_newton, "за", count_newton, "итераций", "с точностью", epsilon)

    sys.stdout = sys.__stdout__
    print("По методу простых итреаций:")
    print("корень =", result_iteration, "за", count_iteration, "итераций", "с точностью", epsilon)
    print("\nПо методу Ньютона:")
    print("корень =", result_newton, "за", count_newton, "итераций", "с точностью", epsilon)