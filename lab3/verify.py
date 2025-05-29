import numpy as np
from pathlib import Path

def verify_results():
    sizes = [100, 500, 1000, 1500, 2000, 2500]
    data_dir = Path("out\\build\\x64-Debug\\files_txt") 
    
    print("Верификация результатов умножения матриц")
    print("----------------------------------------")
    print("Размер | Статус проверки | Макс. расхождение")
    print("-------|-----------------|------------------")
    
    for size in sizes:
        try:
            mat1_path = data_dir / f"mat1_{size}.txt"
            mat2_path = data_dir / f"mat2_{size}.txt"
            result_path = data_dir / f"res_{size}.txt"
            
            A = np.loadtxt(mat1_path)
            B = np.loadtxt(mat2_path)
            C_cpp = np.loadtxt(result_path)
            C_ref = np.matmul(A, B)

            max_diff = np.max(np.abs(C_cpp - C_ref))
            is_correct = np.allclose(C_cpp, C_ref, atol=1e-6)
            
            status = "✓ Успех" if is_correct else f"✗ Ошибка (Δ={max_diff:.2e})"
            print(f"{size}x{size} | {status:^15} | {max_diff:.2e}")
            
        except FileNotFoundError as e:
            print(f"{size}x{size} | Файл не найден | -")
            continue

if __name__ == "__main__":
    verify_results()