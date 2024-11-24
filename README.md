# Supervised Classification

## Overview
This project explores supervised classification by comparing two vectorization
methods and two machine learning models. The goal is to evaluate and analyze
their performance on the DataI_MD dataset.

---

## Prerequisites
Ensure you have the following installed:
- **python** (tested with v3.12.7)
- **pip** (Python package manager)

---

## How to Run

### Step 1: Prepare the Data
Place the dataset DataI_MD.csv in the directory `./data/raw/`.

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
./bob
```

The script will:
1. preprocess the data
2. vectorize the data
3. generate the models from the params acquired in the tuning
4. run the tests for the RQ1 and RQ2

If one step was done, the next time you run the program it will be skiped, but
you can force to run all or individual steps.

For more info about the script
```bash
./bob -h
```

#### 2. Manually
1. preprocess the data
```bash
python ./preprocess.py \
    ./data/raw/DataI_MD.csv \
    -o ./data/dataset/ \
    -s 70/20/10
```
2. vectorize the data
```bash
# using SpaCy
python vectorization.py \
    -i ./data/dataset/train.csv \
    -o ./data/vectorized/spacy/train.csv
# using BoW
## get BoW
python vectorization.py \
    -i ./data/test_dataset/train.csv \
    -o ./data/test_vectorized/bow/BoW.csv \
    -g
python vectorization.py \
    -i ./data/dataset/train.csv \
    -o ./data/vectorized/bow/train.csv \
    -m 'bow' -v ./data/vectorized/bow/BoW.csv
```
3. train and evaluate models
```bash
# RandomForestClassifier
python tuning.py \
    -t ./data/vectorized/spacy/train.csv \
    -d ./data/vectorized/spacy/dev.csv \
    -o ./data/tuning/rforest_params.json \
    -m small -c rforest
# StackingClassifier
python tuning.py \
    -t ./data/test_vectorized/spacy/train.csv \
    -d ./data/test_vectorized/spacy/dev.csv \
    -o ./data/test_tuning/stacking_params.json \
    -m small -c stacking
```
4. generate the models from the params acquired in the train
```bash
# RandomForestClassifier
python inference.py -c rforest \
    ./data/vectorized/spacy/train.csv \
    ./data/vectorized/spacy/dev.csv \
    -i ./data/tuning/rforest_params.json \
    -o ./data/models/rfores.pkl
# StackingClassifier
python inference.py -c stacking \
    ./data/vectorized/spacy/train.csv \
    ./data/vectorized/spacy/dev.csv \
    -i ./data/tuning/rforest_params.json \
    -o ./data/models/stacking.pkl
```
5. run the tests for the RQ1 and RQ2
```bash
# RQ1
python rq1.py \
    -t ./data/test_vectorized/bow/train.csv \
    -d ./data/test_vectorized/bow/dev.csv \
    -o ./data/test_rq1/bow/
python rq1.py \
    -t ./data/test_vectorized/spacy/train.csv \
    -d ./data/test_vectorized/spacy/dev.csv \
    -o ./data/test_rq1/spacy/
# RQ2
python rq2.py \
    ./data/test_vectorized/spacy/train.csv \
    ./data/test_vectorized/spacy/dev.csv \
    -t ./data/test_vectorized/spacy/test.csv \
    -o ./data/test_rq2/rforest/ \
    -p ./data/test_tuning/rforest_params.json \
    -c rforest
python rq2.py \
    ./data/test_vectorized/spacy/train.csv \
    ./data/test_vectorized/spacy/dev.csv \
    -t ./data/test_vectorized/spacy/test.csv \
    -o ./data/test_rq2/stacking/ \
    -p ./data/test_tuning/rforest_params.json \
    -c rforest
```

For more info about the scripts:
```bash
python script.py -h
```

