# Supervised Classification

## Overview
This project explores supervised classification by comparing two vectorization
methods and two machine learning models. The goal is to evaluate and analyze
their performance on a dataset.

---

## Prerequisites
Ensure you have the following installed:
- **python** (tested with v3.12.7)
- **pip** (Python package manager)

---

## How to Run

### Step 1: Prepare the Data
Place the dataset DataI_MD.csv in the directory `./data/raw/`. This step is only
required if you plan to use the `bob` script in [Step 3](#Step-3:-Run).

### Step 2: Set Up the Environment
Create and activate a virtual env., then install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run
#### 1. By using `bob`
Run the main dash script:
```bash
# TODO: provide detailed flags and usage.
./bob 
```

The script will:
- preprocess the data (if not already done)
- vectorize the data (if not already done) (WIP)
- train and evaluate models (WIP)

#### 2. Manually
`TODO`: Provide detailed instructions for manually running each step.

