#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;

void generateMatrix(const string& filename, int rows, int cols) {
    ofstream file(filename);
    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << rand() % 100 << " ";
        }
        file << endl;
    }
}

void readMatrix(const string& filename, vector<vector<int>>& matrix) {
    ifstream file(filename);
    int rows, cols;
    file >> rows >> cols;
    matrix.resize(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    file << matrix.size() << " " << matrix[0].size() << endl;
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            file << value << " ";
        }
        file << endl;
    }
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    
    vector<vector<int>> result(rowsA, vector<int>(colsB, 0));
    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

int main() {
    setlocale(LC_ALL, "RUS");
    const int SIZE = 1000;      
    const int NUM_RUNS = 10;  
    
    generateMatrix("matrixA.txt", SIZE, SIZE);
    generateMatrix("matrixB.txt", SIZE, SIZE);
    
    double totalDuration = 0.0;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        vector<vector<int>> A, B;
        
        readMatrix("matrixA.txt", A);
        readMatrix("matrixB.txt", B);
        
        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> C = multiplyMatrices(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        totalDuration += duration.count();  
        cout << "step " << run + 1 << ": " << duration.count() << " sec" << endl;
        writeMatrix("resultMatrix.txt", C);
    }
    
    double avgTime = totalDuration / NUM_RUNS;

    cout << "Average time to perform matrix multiplication (" << NUM_RUNS << " runs): "
    << avgTime << " seconds" << endl;
    cout << "Matrix size: " << SIZE << "x" << SIZE << ", Total operations: " << SIZE*SIZE*SIZE << endl;
    return 0;
}
