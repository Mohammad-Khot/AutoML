# AutoML Engine

A modular AutoML system designed to automate machine learning workflows, including model selection, training, and benchmarking.

The project focuses on building an extensible AutoML framework capable of running experiments across multiple datasets with minimal configuration.

## Features

* Automatic task and metric inference
* Automated model training and selection
* Multi-dataset experimentation
* OpenML benchmark support
* Reproducible experiments using fixed seeds
* Config-driven pipeline

## Installation

Clone the repository:

```
git clone https://github.com/Mohammad-Khot/AutoML.git
cd AutoML
```

Install dependencies:

```
pip install -r requirements.txt
```

## Running AutoML on Multiple Datasets

Run the built-in multi-dataset experiment suite:

```
python run_multi_dataset.py
```

This will automatically train models on several datasets, including:

* Iris
* Wine
* Breast Cancer
* California Housing
* Synthetic regression dataset

## Running the OpenML Benchmark

To evaluate the engine across multiple real-world datasets:

```
python run_automl_benchmark.py
```

This benchmark:

* downloads datasets from OpenML
* runs AutoML across multiple seeds
* records model performance
* saves results to a CSV file

Benchmark results are stored in:

```
automl_benchmark/benchmark_results.csv
```

## Goals

This project explores building a flexible AutoML framework capable of:

* automating model experimentation
* benchmarking models across datasets
* supporting scalable ML workflows
* enabling reproducible machine learning experiments

## Author

Mohammad Ibrahim Imtiyaz Khot
