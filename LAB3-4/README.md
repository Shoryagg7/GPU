# GPU Computing (UCS635) - Lab 3-4

## Folder Contents
- `assignment3-4.cpp`: OpenMP solutions for Lab Assignment 3 and 4.
- `assignment3-4.exe`: Built executable (generated after compilation).

## Q1: Matrix Multiplication (Serial vs Static vs Dynamic)
Implemented variants:
1. Serial matrix multiplication.
2. OpenMP parallel multiplication with `schedule(static)`.
3. OpenMP parallel multiplication with `schedule(dynamic, chunk)`.

Benchmark behavior:
- Runs practical matrix sizes by default: 256, 512, 1024.
- Compares execution time and numerical differences (`StaticDiff`, `DynamicDiff`) against serial output.

Optional dataset mode for 10000x10000 matrices:
- Program supports file-based input mode through command line:
```bash
./assignment3-4.exe --run10000 <matrixA_file> <matrixB_file>
```
- Each file must contain exactly 10000 x 10000 numeric values in row-major order.
- In this mode, static and dynamic parallel runs are timed (serial is skipped due very high runtime).

## Q2: Series of 2 and 4 on Different Threads
- Uses OpenMP sections with 2 threads.
- One thread prints series of 2; the other prints series of 4.
- Execution times for each section are reported.

## Q3: Concurrent Sum and Synchronization
Implements and compares:
1. Unsynchronized parallel sum (race condition demonstration).
2. `critical` synchronized sum.
3. `atomic` synchronized sum.
4. Serial reference sum.

Reports:
- Correctness of computed sums.
- Execution times for each approach.

## Build and Run (Windows, g++)
```bash
g++ -fopenmp assignment3-4.cpp -o assignment3-4.exe
./assignment3-4.exe
```

Run dataset mode (Q1, 10000x10000):
```bash
./assignment3-4.exe --run10000 matrixA.txt matrixB.txt
```

## Notes
- Output order from threads can vary between runs.
- `critical` is generally slower than `atomic` for simple increments.
- Static scheduling usually performs well for uniform dense matrix workloads.
