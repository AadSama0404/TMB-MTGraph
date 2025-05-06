# TMB-MTGraph

This repository provides the implementation of **TMB-MTGraph**, a prognostic prediction model based on clonal mutation features. It is designed to handle heterogeneous cancer cohorts and improve survival prediction using clonal architecture-aware learning.

## Directory Structure

- `data/`  
  Contains raw data and the preprocessing script.
  
- `baseline/`  
  Includes all baseline models used for comparison.

- `results/`  
  Stores outputs such as predictions and plots.

- `run_TMB_MTGraph.py`  
  Main script to train and evaluate TMB-MTGraph.

## Getting Started

### Step 1: Data Preprocessing

Run the following command to preprocess input data:

```bash
python data/data_preprocessing.py
```

### Step 2: Run TMB-MTGraph

Execute the main script to train and evaluate the model:

```bash
python run_TMB_MTGraph.py
```

Outputs including survival analysis results and plots will be saved in the results/ folder.

### Step 3: Run Baseline Models

All baseline methods are located in the baseline/ folder. You can run them individually for performance comparison.
