# Parallel Recognizer

This module implements the core parallel face recognition engine. It distributes images across multiple processes and dynamically adjusts worker allocation to maximise CPU utilisation.

## Features

- Dynamic worker selection  
- Adaptive chunk sizing  
- Load balancing using work-stealing  
- Parallel encoding and matching  
- Internal performance monitoring  

## API Reference

::: src.recognizer