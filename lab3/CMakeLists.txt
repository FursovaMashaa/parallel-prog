cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplicationMPI LANGUAGES CXX)

# ��������� ��������� C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ����� MPI (��� Linux ����� WSL)
find_package(MPI REQUIRED)

# �������� ������������ �����
add_executable(generate_write
        "main.cpp"  # �������� �� ��� �������� ����
)

# ����������� MPI
target_link_libraries(generate_write
        PRIVATE
        MPI::MPI_CXX
)

# �������������� ����� ����������
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(main PRIVATE -Wall -Wextra)
endif()