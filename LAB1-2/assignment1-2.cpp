#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <omp.h>

using namespace std;

void question1_and_2_notes()
{
    cout << "================ Q1 & Q2 =================\n";
    cout << "Q1) OpenMP enables advanced parallel computing on shared-memory systems using\n";
    cout << "    APIs for thread creation, work sharing, synchronization, and data scoping.\n\n";

    cout << "Q2) Computational power is commonly measured in FLOPS (Floating Point Operations\n";
    cout << "    Per Second), typically reported as GFLOPS/TFLOPS.\n";

    // Detected from this node: 11th Gen Intel i7-11370H, 4 physical cores, 3.302 GHz max.
    const int physicalCores = 4;
    const double maxFrequencyGHz = 3.302;

    // Assumption: 16 FP32 operations/cycle/core (AVX2 + FMA, conservative estimate).
    const double fp32OpsPerCyclePerCore = 16.0;
    const double estimatedPeakGFLOPS = physicalCores * maxFrequencyGHz * fp32OpsPerCyclePerCore;

    cout << fixed << setprecision(2);
    cout << "    Formula used: Peak FLOPS = Cores x Frequency(Hz) x FLOPs/Cycle\n";
    cout << "    Estimated Peak FP32 = " << physicalCores << " x " << maxFrequencyGHz
         << " x " << fp32OpsPerCyclePerCore << " = " << estimatedPeakGFLOPS
         << " GFLOPS\n\n";
}

void question3_family_names_from_threads()
{
    cout << "================ Q3 ======================\n";
    vector<string> familyNames = {
        "Father", "Mother", "Brother", "Sister", "Grandfather", "Grandmother", "Uncle", "Aunt"};

#pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        int totalThreads = omp_get_num_threads();

        string name = (tid < static_cast<int>(familyNames.size())) ? familyNames[tid] : "FamilyMember";

#pragma omp critical
        {
            cout << "Thread/Job ID: " << tid << " out of " << totalThreads
                 << " -> " << name << "\n";
        }
    }
    cout << "\n";
}

void question4_sum_of_squares_of_thread_ids()
{
    cout << "================ Q4 ======================\n";
    int sumOfSquares = 0;

#pragma omp parallel num_threads(8) reduction(+ : sumOfSquares)
    {
        int tid = omp_get_thread_num();
        int sq = tid * tid;
        sumOfSquares += sq;

#pragma omp critical
        {
            cout << "Thread ID " << tid << " square = " << sq << "\n";
        }
    }

    cout << "Sum of squares of thread IDs = " << sumOfSquares << "\n\n";
}

void question5_private_aryabhatta()
{
    cout << "================ Q5 ======================\n";
    int Aryabhatta = 10;

#pragma omp parallel num_threads(8) private(Aryabhatta)
    {
        // private variable is uninitialized per-thread, so set it explicitly.
        Aryabhatta = 10;
        int tid = omp_get_thread_num();
        int result = tid * Aryabhatta;

#pragma omp critical
        {
            cout << "Thread ID " << tid << ": " << tid << " x " << Aryabhatta
                 << " = " << result << "\n";
        }
    }
    cout << "\n";
}

void question6_partial_sum_with_lastprivate()
{
    cout << "================ Q6 ======================\n";
    const int N = 20;
    int totalSum = 0;
    int lastProcessedValue = 0;

#pragma omp parallel for reduction(+ : totalSum) lastprivate(lastProcessedValue)
    for (int i = 1; i <= N; ++i)
    {
        int tid = omp_get_thread_num();
        totalSum += i;
        lastProcessedValue = i;

#pragma omp critical
        {
            cout << "Thread " << tid << " processed value " << i << "\n";
        }
    }

    cout << "Lastprivate value (last loop iteration) = " << lastProcessedValue << "\n";
    cout << "Total sum of first " << N << " natural numbers = " << totalSum << "\n";
    cout << "Expected sum = " << (N * (N + 1)) / 2 << "\n\n";
}

int main()
{
    cout << "OpenMP Lab Assignment-1 Solutions\n\n";

    question1_and_2_notes();
    question3_family_names_from_threads();
    question4_sum_of_squares_of_thread_ids();
    question5_private_aryabhatta();
    question6_partial_sum_with_lastprivate();

    return 0;
}
