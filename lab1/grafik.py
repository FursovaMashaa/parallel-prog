import matplotlib.pyplot as plt
import numpy as np

matrix_sizes = [
    'Матрица 10x10', 'Матрица 100x100', 'Матрица 500x500',
    'Матрица 1000x1000', 'Матрица 1500x1500', 'Матрица 2000x2000',
    'Матрица 2500x2500'
]
mean_times = [
    4.192e-05, 0.0262727, 3.49101,
    35.9732, 120.822, 581.747, 1104.99
]

sizes = []
for size in matrix_sizes:
    n = int(size.split('x')[0].split()[-1])
    sizes.append(n)

plt.figure(figsize=(10, 6))
plt.plot(sizes, mean_times, marker='o', linestyle='-', color='b')
plt.title("Зависимость среднего времени вычислений от размера матрицы")
plt.xlabel("Размер матрицы (n x n)")
plt.ylabel("Среднее время вычисления (секунды)")
plt.grid(True)

plt.savefig('result.png', dpi=300, bbox_inches='tight')
plt.show()
