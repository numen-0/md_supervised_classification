# Supervised Classification

## Overview
This project explores supervised classification by comparing two vectorization
methods and two machine learning models. The goal is to evaluate and analyze
their performance on the DataI_MD dataset.

---

## Prerequisites
Ensure you have the following installed:
- **python** (tested with v3.12.7).
- **pip** (Python package manager).
- you have access to the DataI_MD dataset.

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
Run the main dash script, this will take a lot of time due to `tuning.py`:
```bash
# +12h to run all
./bob
```

If you only want a quick test:
```bash
# ~30min to run all
./bob -t
```

You can add the flag `-s` to silence some output:
```bash
./bob -t -s
```

If you want to skip tuning:
```bash
# for full
cp -r ./example_params ./data/tuning
./bob

# for test
cp -r ./example_params ./data/test_tuning
./bob -t
```

The script will:
1. `preprocess.py`: preprocess the data
2. `vectorization.py`: vectorize the data (BoW + SpaCy)
3. `tuning.py`: search for best params for both models (tuning)
4. `train.py`: generate the models from the params adquired in the tuning
5. `rqextra.py`: run the tests for the RQext
6. `rq1.py`: run the tests for the RQ1
7. `rq2_1.py` + `rq2_2.py`: run the tests for the RQ2

If one step was done, the next time you run the program it will be skiped, but
you can force to run all or individual steps.

For more info about the script
```bash
./bob -h
```

#### 2. Manually
NOTE: if you want to skip the tuning, there are some params in the
`./example_params`. If you want to run `./rq2.py` they are necesary.

NOTE: for more info about (almost) any scripts.py:
```bash
python script.py -h
```

##### preprocess the data
```bash
python ./preprocess.py \
    ./data/raw/DataI_MD.csv \
    -o ./data/dataset/ \
    -s 70/20/10
```

##### vectorize the data
```bash
# using BoW
## get BoW
python vectorization.py \
    -i ./data/dataset/train.csv \
    -o ./data/vectorized/bow/BoW.csv \
    -g
## train
python vectorization.py \
    -i ./data/dataset/train.csv \
    -o ./data/vectorized/bow/train.csv \
    -m 'bow' -v ./data/vectorized/bow/BoW.csv
## ...
# using SpaCy
## train
python vectorization.py \
    -i ./data/dataset/train.csv \
    -o ./data/vectorized/spacy/train.csv \
    -m 'spacy'
## ...
```

##### train and evaluate models
```bash
# RandomForestClassifier
python tuning.py \
    -t ./data/vectorized/spacy/train.csv \
    -d ./data/vectorized/spacy/dev.csv \
    -o ./data/tuning/rforest_params.json \
    -c rforest -m small
# ...
```

##### generate the models from the params acquired in the train
```bash
# RandomForestClassifier
python inference.py \
    ./data/vectorized/spacy/train.csv \
    ./data/vectorized/spacy/dev.csv \
    -i ./data/tuning/rforest_params.json \
    -o ./data/models/rfores.pkl \
    -c rforest
# ...
```

##### run the tests for the RQext, RQ1 and RQ2
```bash
# RQext
python rqextra.py \
    -t ./data/vectorized/spacy/test.csv \
    -o ./data/rqext/results.txt
# RQ1
python rq1.py \
    -t ./data/vectorized/bow/train.csv \
    -d ./data/vectorized/bow/dev.csv \
    -o ./data/rq1/bow/
python rq1.py \
    -t ./data/vectorized/spacy/train.csv \
    -d ./data/vectorized/spacy/dev.csv \
    -o ./data/rq1/spacy/
# RQ2
## RandomForestClassifier
python rq2_1.py \
    ./data/vectorized/spacy/train.csv \
    ./data/vectorized/spacy/dev.csv \
    -t ./data/vectorized/spacy/test.csv \
    -o ./data/rq2/rforest/ \
    -p ./data/tuning/rforest_params.json \
    -c rforest
## ...
python rq2_2.py \
        ./data/vectorized/spacy/train.csv \
        ./data/vectorized/spacy/dev.csv \
        -t ./data/vectorized/spacy/test.csv \
        -A ./data/vectorized/spacy/rforest_params.json \
        -B ./data/vectorized/spacy/gradient_params.json \
        -C ./data/vectorized/spacy/svc_params.json \
        -D ./data/vectorized/spacy/stacking_params.json \
        -o ./data/rq2/extra/
```
