# Parallel Face Recognition System

This project implements a high-performance face recognition system using both **serial and parallel processing techniques**. The main objective is to demonstrate how parallel computing can significantly improve execution time when performing face matching on large image datasets.

The system uses the `face_recognition` library for facial encoding and comparison, and applies **multiprocessing with dynamic worker allocation** to distribute the workload efficiently across available CPU cores.

Key features of this project include:

- Serial face recognition for baseline performance comparison  
- Parallel face recognition using multiprocessing  
- Intelligent dynamic worker allocation based on dataset size  
- Automatic chunk size optimisation  
- Load balancing using a work-stealing strategy  
- Detailed performance statistics and speedup analysis  
