Code associated with ICLR 2021 paper: Mahto, S., Vo, V.A., Turek, J.S., Huth, A. "Multi-timescale representation learning in LSTM language models."

This is adapted from the AWD-LSTM-LM code available here: [https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm). To more closely reproduce their results with Pytorch 0.4, see the `legacy` folder.

# Commands for training a multi-timescale (MTS) language model 
## Required dependencies: Python3.6 or above, Numpy, Scipy and Pytorch1.7.0 or above with CUDA version 10.1 

### Example script to train and evaluate a standard and MTS LM on PTB dataset:

```shell
bash run.sh
```

### Detailed description:

### 1. To download PTB/WIKI data:

```shell
bash getdata.sh
```

### 2. model_mts.py defines the multi-timescale language model.

### 3. To train a multi-timescale model, use train_mts.py as follows:

#### On PTB data

```python
python train_mts.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1000 --save train_mts.pt 
```

#### On Wiki data 

```python
python train_mts.py --data data/wikitext-2 --dropouth 0.2 --seed 1882 --epoch 1000 --save train_mts.pt 
```

### 4. To evaluate model on test set: including different word frequency bins and bootstrap test set 

#### Trained LM on PTB data: 
```python
python model_evaluation.py --model_name train_mts.pt --data data/penn/
```

#### Trained LM on Wiki data: 
```python
python model_evaluation.py --model_name train_mts.pt --data data/wikitext-2/
```


# Formal Language: Dyck-2 Grammar
 
## Creating the dataset:

```python
python create_dyckn.py 2 -p 0.25 0.25 -q 0.25 --train 10000 --validation 2000 --test 5000 --max_length 200
```

The option `--jobs <num_cores>` allows to parallelize and generate the dataset faster.  

## Training the models:

To train the models use the command: 
```python
python run_dyckn.py -u 256 -l 1 --epochs 2000 -s 200 --lr 1e-4  --model MTS
```

Use `--model MTS` for the multi-timescale LSTM model and `--model Baseline` for the baseline LSTM model.
