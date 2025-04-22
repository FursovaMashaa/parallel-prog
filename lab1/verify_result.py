import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().strip().split())
        matrix = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            matrix[i] = list(map(int, f.readline().strip().split()))
    return matrix

A = read_matrix("matrixA.txt")
B = read_matrix("matrixB.txt")
C_computed = read_matrix("resultMatrix.txt")

C_expected = np.dot(A, B)

if np.array_equal(C_computed, C_expected):
    print("Результаты совпадают. Умножение выполнено корректно.")
else:
    print("Результаты не совпадают. Умножение выполнено некорректно.")
