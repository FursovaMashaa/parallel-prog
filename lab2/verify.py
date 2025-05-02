import numpy as np
from pathlib import Path

def verify_results():
    sizes = [100, 500, 1000, 2000, 2500]
    data_dir = Path("matrix_data")
    result_dir = Path("results")
    
    print("Проверка результатов:")
    print("---------------------")
    print("Размер | Потоки | Статус")
    
    for size in sizes:
        A = np.loadtxt(data_dir / f"matrixA_{size}.txt", skiprows=1)
        B = np.loadtxt(data_dir / f"matrixB_{size}.txt", skiprows=1)
        C_cpp = np.loadtxt(result_dir / f"result_{size}.txt", skiprows=1)
        C_ref = A @ B
        
        max_diff = np.max(np.abs(C_cpp - C_ref))
        is_correct = np.allclose(C_cpp, C_ref, atol=1e-6)
        
        print(f"{size}x{size} | {15:^7} | {'Результат верный' if is_correct else 'Результат неверный'}")

if __name__ == "__main__":
    verify_results()