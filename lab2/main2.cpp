#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <iomanip>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

void createDirectory(const string& dirname) {
    if (!fs::exists(dirname)) {
        fs::create_directory(dirname);
    }
}

void generateMatrix(const string& filename, int rows, int cols) {
    ofstream file(filename);
    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        string row;
        for (int j = 0; j < cols; ++j) {
            row += to_string(rand() % 100) + " ";
        }
        file << row << endl;
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
    int rows = matrix.size();
    int cols = matrix[0].size();
    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B, int numThreads) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    
    vector<vector<int>> result(rowsA, vector<int>(colsB, 0));
    
    omp_set_num_threads(numThreads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

void writeResultsToTxt(const string& filename, 
                      const vector<vector<double>>& results,
                      const vector<int>& sizes,
                      const vector<int>& threadCounts,
                      int numRuns) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error opening results file!" << endl;
        return;
    }
    
    out << "============================================\n";
    out << "      MATRIX MULTIPLICATION BENCHMARK       \n";
    out << "============================================\n\n";
    
    out << "Matrix sizes: ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        out << sizes[i] << "x" << sizes[i];
        if (i != sizes.size() - 1) out << ", ";
    }
    out << "\nThread counts: ";
    for (size_t i = 0; i < threadCounts.size(); ++i) {
        out << threadCounts[i];
        if (i != threadCounts.size() - 1) out << ", ";
    }
    out << "\nNumber of runs per test: " << numRuns << "\n\n";
    
    out << "|  Matrix Size |   Threads  |  Avg Time (sec)  |  Operations  |\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        for (size_t j = 0; j < threadCounts.size(); ++j) {
            long operations = sizes[i] * sizes[i] * sizes[i];
            out << "| " << setw(10) << sizes[i] << "x" << sizes[i] << " | "
                << setw(7) << threadCounts[j] << " | "
                << setw(15) << fixed << setprecision(6) << results[i][j] << " | "
                << setw(14) << operations << " |\n";
        }
    }
    out.close();
}

int main() {
    setlocale(LC_ALL, "RUS");
    
    const vector<int> sizes = {100, 500, 1000, 2000, 2500};
    const vector<int> threadCounts = {2, 5, 10, 15};
    const int NUM_RUNS = 10;
    const string DATA_DIR = "matrix_data";
    const string RESULT_DIR = "results";
    const string RESULT_FILE_TXT = RESULT_DIR + "/results.txt";
    
    createDirectory(DATA_DIR);
    createDirectory(RESULT_DIR);
    
    vector<vector<double>> performanceResults(sizes.size(), vector<double>(threadCounts.size()));

    for (size_t sizeIdx = 0; sizeIdx < sizes.size(); ++sizeIdx) {
        int size = sizes[sizeIdx];
        string matrixA_file = DATA_DIR + "/matrixA_" + to_string(size) + ".txt";
        string matrixB_file = DATA_DIR + "/matrixB_" + to_string(size) + ".txt";
        string result_file = RESULT_DIR + "/result_" + to_string(size) + ".txt";
        
        ifstream f(matrixA_file);
        if (!f.good()) {
            generateMatrix(matrixA_file, size, size);
            generateMatrix(matrixB_file, size, size);
        }
        
        for (size_t threadIdx = 0; threadIdx < threadCounts.size(); ++threadIdx) {
            int numThreads = threadCounts[threadIdx];
            double totalDuration = 0.0;

            for (int run = 0; run < NUM_RUNS; ++run) {
                vector<vector<int>> A, B;
                readMatrix(matrixA_file, A);
                readMatrix(matrixB_file, B);
                
                auto start = chrono::high_resolution_clock::now();
                vector<vector<int>> C = multiplyMatrices(A, B, numThreads);
                auto end = chrono::high_resolution_clock::now();
                
                chrono::duration<double> duration = end - start;
                totalDuration += duration.count();
                
                if (run == NUM_RUNS - 1) {
                    writeMatrix(result_file, C);
                }
                
                cout << "Size: " << size << " Threads: " << numThreads 
                     << " Step " << run + 1 << ": " << duration.count() << " sec" << endl;
            }

            double avgTime = totalDuration / NUM_RUNS;
            performanceResults[sizeIdx][threadIdx] = avgTime;
            
            cout << "Average time for size " << size << " with " << numThreads 
                 << " threads: " << fixed << setprecision(4) << avgTime << " sec" << endl;
        }
    }

    writeResultsToTxt(RESULT_FILE_TXT, performanceResults, sizes, threadCounts, NUM_RUNS);

    cout << "DONE " << endl;
    return 0;
}