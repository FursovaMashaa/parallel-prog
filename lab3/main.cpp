#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>
#include <mpi.h>
#include <filesystem>

using namespace std;

vector<vector<int>> create_matrix(int dim, int rank_id) {
    vector<vector<int>> mat(dim, vector<int>(dim));
    mt19937 generator(static_cast<unsigned int>(time(nullptr)) + rank_id);
    uniform_int_distribution<int> distribution(0, 99);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            mat[i][j] = distribution(generator);
    return mat;
}

void write_matrix(const vector<vector<int>>& mat, const string& filename) {
    ofstream fout(filename);
    for (const auto& row : mat) {
        for (int val : row)
            fout << val << ' ';
        fout << '\n';
    }
}

vector<vector<int>> read_matrix(const string& filename) {
    ifstream fin(filename);
    vector<vector<int>> mat;
    string line;
    while (getline(fin, line)) {
        istringstream stream(line);
        vector<int> row;
        int x;
        while (stream >> x)
            row.push_back(x);
        if (!row.empty())
            mat.push_back(row);
    }
    return mat;
}

vector<vector<int>> multiply_parallel(const vector<vector<int>>& A, const vector<vector<int>>& B, int rank_id, int total_ranks) {
    int n = static_cast<int>(A.size());
    int step = n / total_ranks;
    int start = rank_id * step;
    int finish = (rank_id == total_ranks - 1) ? n : start + step;

    vector<vector<int>> local_matrix(n, vector<int>(n, 0));
    for (int i = start; i < finish; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                local_matrix[i][j] += A[i][k] * B[k][j];

    vector<int> flat_local(n * n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat_local[i * n + j] = local_matrix[i][j];

    vector<int> flat_final(n * n, 0);
    MPI_Reduce(flat_local.data(), flat_final.data(), n * n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    vector<vector<int>> output;
    if (rank_id == 0) {
        output.resize(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                output[i][j] = flat_final[i * n + j];
    }

    return output;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank_id, total_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    vector<int> sizes = {100, 500, 1000, 1500, 2000, 2500};
    vector<double> durations(sizes.size(), 0.0);
    string folder = "files_txt/";

    if (rank_id == 0) {
        std::filesystem::create_directory(folder);
        for (int sz : sizes) {
            auto mat1 = create_matrix(sz, rank_id);
            auto mat2 = create_matrix(sz, rank_id);
            write_matrix(mat1, folder + "mat1_" + to_string(sz) + ".txt");
            write_matrix(mat2, folder + "mat2_" + to_string(sz) + ".txt");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        int dim = sizes[idx];
        string fileA = folder + "mat1_" + to_string(dim) + ".txt";
        string fileB = folder + "mat2_" + to_string(dim) + ".txt";
        string output_file = folder + "res_" + to_string(dim) + ".txt";

        vector<vector<int>> matA, matB;

        if (rank_id == 0) {
            matA = read_matrix(fileA);
            matB = read_matrix(fileB);

            for (int p = 1; p < total_ranks; ++p)
                MPI_Send(&dim, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

            for (int p = 1; p < total_ranks; ++p) {
                for (const auto& row : matA)
                    MPI_Send(row.data(), dim, MPI_INT, p, 1, MPI_COMM_WORLD);
                for (const auto& row : matB)
                    MPI_Send(row.data(), dim, MPI_INT, p, 2, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matA.resize(dim, vector<int>(dim));
            matB.resize(dim, vector<int>(dim));

            for (auto& row : matA)
                MPI_Recv(row.data(), dim, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (auto& row : matB)
                MPI_Recv(row.data(), dim, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank_id == 0)
            cout << "Processing matrix size " << dim << "x" << dim << "\n";

        auto t0 = chrono::steady_clock::now();
        auto result = multiply_parallel(matA, matB, rank_id, total_ranks);
        auto t1 = chrono::steady_clock::now();

        if (rank_id == 0) {
            write_matrix(result, output_file);
            durations[idx] = chrono::duration<double>(t1 - t0).count();
        }
    }

    if (rank_id == 0) {
        ofstream log("results.txt");
        log << "Timing (s):\n";
        for (size_t i = 0; i < sizes.size(); ++i)
            log << sizes[i] << ": " << durations[i] << " s\n";
    }

    MPI_Finalize();
    return 0;
}
