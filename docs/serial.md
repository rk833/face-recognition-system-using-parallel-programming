# Serial Implementation

The serial implementation processes images one at a time using a single CPU core. It is used to provide a baseline for evaluating the benefits of parallel execution.

Two serial approaches exist in this project:

- Standalone serial program: `task1_4_serial.py`  
- Integrated serial comparison inside `main.py`  

::: main.run_serial_comparison