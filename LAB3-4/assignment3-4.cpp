#include <omp.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;

static inline int idx(int r, int c, int n)
{
    return r * n + c;
}

vector<double> make_random_matrix(int n, int seed)
{
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<double> mat(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (double &v : mat)
    {
        v = dist(gen);
    }
    return mat;
}

void multiply_serial(const vector<double> &a, const vector<double> &b, vector<double> &c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[idx(i, k, n)] * b[idx(k, j, n)];
            }
            c[idx(i, j, n)] = sum;
        }
    }
}

void multiply_omp_static(const vector<double> &a, const vector<double> &b, vector<double> &c, int n)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[idx(i, k, n)] * b[idx(k, j, n)];
            }
            c[idx(i, j, n)] = sum;
        }
    }
}

void multiply_omp_dynamic(const vector<double> &a, const vector<double> &b, vector<double> &c, int n, int chunk)
{
#pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[idx(i, k, n)] * b[idx(k, j, n)];
            }
            c[idx(i, j, n)] = sum;
        }
    }
}

double max_abs_diff(const vector<double> &x, const vector<double> &y)
{
    double best = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        best = max(best, fabs(x[i] - y[i]));
    }
    return best;
}

bool load_matrix_from_file(const string &filePath, int n, vector<double> &out)
{
    ifstream in(filePath);
    if (!in.is_open())
    {
        return false;
    }

    out.assign(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (!(in >> out[i]))
        {
            return false;
        }
    }
    return true;
}

void run_q1_matrix_multiplication_benchmark()
{
    cout << "================ Q1 ======================\n";
    cout << "Matrix Multiplication: Serial vs OpenMP Static vs OpenMP Dynamic\n";
    cout << "Threads available: " << omp_get_max_threads() << "\n\n";

    vector<int> sizes = {256, 512, 1024};
    const int dynamicChunk = 8;

    cout << left << setw(10) << "N"
         << setw(16) << "Serial(s)"
         << setw(16) << "Static(s)"
         << setw(16) << "Dynamic(s)"
         << setw(16) << "StaticDiff"
         << setw(16) << "DynamicDiff" << "\n";

    for (int n : sizes)
    {
        vector<double> a = make_random_matrix(n, 101 + n);
        vector<double> b = make_random_matrix(n, 202 + n);

        vector<double> cSerial(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
        vector<double> cStatic(cSerial.size(), 0.0);
        vector<double> cDynamic(cSerial.size(), 0.0);

        double t0 = omp_get_wtime();
        multiply_serial(a, b, cSerial, n);
        double t1 = omp_get_wtime();

        double t2 = omp_get_wtime();
        multiply_omp_static(a, b, cStatic, n);
        double t3 = omp_get_wtime();

        double t4 = omp_get_wtime();
        multiply_omp_dynamic(a, b, cDynamic, n, dynamicChunk);
        double t5 = omp_get_wtime();

        double diffStatic = max_abs_diff(cSerial, cStatic);
        double diffDynamic = max_abs_diff(cSerial, cDynamic);

        cout << left << setw(10) << n
             << setw(16) << fixed << setprecision(4) << (t1 - t0)
             << setw(16) << fixed << setprecision(4) << (t3 - t2)
             << setw(16) << fixed << setprecision(4) << (t5 - t4)
             << setw(16) << scientific << setprecision(2) << diffStatic
             << setw(16) << scientific << setprecision(2) << diffDynamic << "\n";
    }

    cout << "\nDiscussion:\n";
    cout << "1) Static scheduling has lower runtime overhead and is usually faster when workload is uniform.\n";
    cout << "2) Dynamic scheduling balances uneven workloads better but has extra scheduling overhead.\n";
    cout << "3) For dense matrix multiplication, static often wins because each row has similar cost.\n";
    cout << "4) For very large matrices (e.g., 10000x10000), use blocked algorithms and enough RAM.\n\n";
}

void run_q1_dataset_10000_benchmark(const string &matrixAPath, const string &matrixBPath)
{
    cout << "Q1 Dataset Mode (10000x10000)\n";
    const int n = 10000;
    const int dynamicChunk = 8;

    vector<double> a;
    vector<double> b;
    vector<double> c(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);

    cout << "Loading matrix A from: " << matrixAPath << "\n";
    cout << "Loading matrix B from: " << matrixBPath << "\n";

    bool okA = load_matrix_from_file(matrixAPath, n, a);
    bool okB = load_matrix_from_file(matrixBPath, n, b);

    if (!okA || !okB)
    {
        cout << "Dataset loading failed. Ensure both files contain exactly 10000x10000 numeric values.\n\n";
        return;
    }

    cout << "Dataset loaded successfully. Running parallel static and dynamic only (serial skipped due very high cost).\n";

    double t0 = omp_get_wtime();
    multiply_omp_static(a, b, c, n);
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    multiply_omp_dynamic(a, b, c, n, dynamicChunk);
    double t3 = omp_get_wtime();

    cout << fixed << setprecision(4);
    cout << "Static schedule time (s):  " << (t1 - t0) << "\n";
    cout << "Dynamic schedule time (s): " << (t3 - t2) << "\n\n";
}

void print_series_by_thread(int base, int terms, int tid)
{
    cout << "Thread " << tid << " printing series of " << base << ": ";
    for (int i = 1; i <= terms; ++i)
    {
        cout << (base * i);
        if (i < terms)
        {
            cout << ' ';
        }
    }
    cout << "\n";
}

void run_q2_series_two_and_four()
{
    cout << "================ Q2 ======================\n";
    const int terms = 20;
    double tStart = omp_get_wtime();
    double tSeries2 = 0.0;
    double tSeries4 = 0.0;

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            int tid = omp_get_thread_num();
            double t0 = omp_get_wtime();
#pragma omp critical
            {
                print_series_by_thread(2, terms, tid);
            }
            tSeries2 = omp_get_wtime() - t0;
        }

#pragma omp section
        {
            int tid = omp_get_thread_num();
            double t0 = omp_get_wtime();
#pragma omp critical
            {
                print_series_by_thread(4, terms, tid);
            }
            tSeries4 = omp_get_wtime() - t0;
        }
    }

    double totalTime = omp_get_wtime() - tStart;
    cout << fixed << setprecision(6);
    cout << "Series of 2 time (s): " << tSeries2 << "\n";
    cout << "Series of 4 time (s): " << tSeries4 << "\n";
    cout << "Total parallel section time (s): " << totalTime << "\n\n";
}

long long sum_serial(const vector<int> &arr)
{
    long long sum = 0;
    for (int v : arr)
    {
        sum += v;
    }
    return sum;
}

long long sum_unsynchronized(const vector<int> &arr)
{
    long long totalSum = 0;

// Intentional race condition for demonstration.
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(arr.size()); ++i)
    {
        totalSum += arr[i];
    }
    return totalSum;
}

long long sum_with_critical(const vector<int> &arr)
{
    long long totalSum = 0;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(arr.size()); ++i)
    {
#pragma omp critical
        {
            totalSum += arr[i];
        }
    }
    return totalSum;
}

long long sum_with_atomic(const vector<int> &arr)
{
    long long totalSum = 0;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(arr.size()); ++i)
    {
#pragma omp atomic
        totalSum += arr[i];
    }
    return totalSum;
}

void run_q3_shared_sum_synchronization()
{
    cout << "================ Q3 ======================\n";
    const int n = 8'000'000;
    vector<int> arr(n, 1);

    double t0 = omp_get_wtime();
    long long sSerial = sum_serial(arr);
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    long long sUnsync = sum_unsynchronized(arr);
    double t3 = omp_get_wtime();

    double t4 = omp_get_wtime();
    long long sCritical = sum_with_critical(arr);
    double t5 = omp_get_wtime();

    double t6 = omp_get_wtime();
    long long sAtomic = sum_with_atomic(arr);
    double t7 = omp_get_wtime();

    cout << "Expected sum = " << n << "\n";
    cout << "Serial sum   = " << sSerial << " , time(s) = " << fixed << setprecision(6) << (t1 - t0) << "\n";
    cout << "Unsync sum   = " << sUnsync << " , time(s) = " << fixed << setprecision(6) << (t3 - t2)
         << " (race condition possible)\n";
    cout << "Critical sum = " << sCritical << " , time(s) = " << fixed << setprecision(6) << (t5 - t4) << "\n";
    cout << "Atomic sum   = " << sAtomic << " , time(s) = " << fixed << setprecision(6) << (t7 - t6) << "\n\n";

    cout << "Issue identified: Concurrent updates to shared total_sum cause data races in unsynchronized code.\n";
    cout << "Observation: critical is correct but often slower due to lock contention; atomic is usually faster than critical for simple increments.\n\n";
}

int main(int argc, char *argv[])
{
    cout << "OpenMP Lab Assignment 3-4 Solutions\n\n";

    run_q1_matrix_multiplication_benchmark();

    if (argc >= 4 && string(argv[1]) == "--run10000")
    {
        run_q1_dataset_10000_benchmark(argv[2], argv[3]);
    }
    else
    {
        cout << "Tip: To benchmark provided 10000x10000 dataset files, run:\n";
        cout << "assignment3-4.exe --run10000 <matrixA_file> <matrixB_file>\n\n";
    }

    run_q2_series_two_and_four();
    run_q3_shared_sum_synchronization();

    return 0;
}
