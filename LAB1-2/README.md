# GPU Computing (UCS635) - Lab 1-2

## Folder Contents
- `assignment1-2.cpp`: Combined implementation for Lab Assignment 1 and 2 questions.

## Topics Covered
1. OpenMP advanced environment and APIs (overview notes).
2. Computational power metric (FLOPS/GFLOPS) with sample system estimate.
3. Printing family member names from different OpenMP threads with thread/job IDs.
4. Printing square of each thread ID and total sum of squares.
5. Using a private variable (`Aryabhatta = 10`) and multiplying it with thread ID.
6. Parallel sum of first 20 natural numbers using `lastprivate` (plus reduction for safe accumulation).

## Build and Run (Windows, g++)
```bash
g++ -fopenmp assignment1-2.cpp -o assignment1-2.exe
./assignment1-2.exe
```

## Expected Output Highlights
- Thread-wise output order may vary between runs (normal in parallel programs).
- Q4 prints each thread square and final sum.
- Q5 confirms private variable usage per thread.
- Q6 prints:
  - lastprivate value from final loop iteration
  - total sum of first 20 natural numbers (= 210)

## Notes
- You can change thread count by editing `num_threads(...)` in the source.
- Keep OpenMP enabled during compilation (`-fopenmp`).
