# Serial Implementation

The serial implementation processes images one at a time using a single CPU core. It is used to provide a baseline for evaluating the benefits of parallel execution.

Two serial approaches exist in this project:

- Standalone serial program: `task1_4_serial.py`  
- Integrated serial comparison inside `task_1_4_main.py`  

## Purpose

- Establish baseline execution time  
- Validate recognition correctness  
- Calculate speedup and efficiency  

## API Reference (integrated serial mode)

::: task_1_4_main.run_serial_comparison