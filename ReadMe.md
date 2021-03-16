Code associated with ICLR 2021 paper: Mahto, S., Vo, V.A., Turek, J.S., Huth, A. "Multi-timescale representation learning in LSTM language models."

This is adapted from the AWD-LSTM-LM code available here: [https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

# Commands for training a multi-timescale (MTS) language model 
## Required dependencies: Python3.6 or above, Numpy, Scipy and Pytorch1.7.0 or above with CUDA version 10.1 

### Example script to train and evaluate a standard and MTS LM on PTB dataset:

bash run.sh

### Detailed description:

### 1. To download PTB/WIKI data:

bash getdata.sh

### 2. model_mts.py defines the multi-timescale language model.

### 3. To train a multi-timescale model, use train_mts.py as follows:

#### On PTB data

python train_mts.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1000 --save train_mts.pt 

#### On Wiki data 

python train_mts.py --data data/wikitext-2 --dropouth 0.2 --seed 1882 --epoch 1000 --save train_mts.pt 

### 4. To evaluate model on test set: including different word frequency bins and bootstrap test set 

#### Trained LM on PTB data: 
python model_evaluation.py --model_name train_mts.pt --data data/penn/

#### Trained LM on Wiki data: 
python model_evaluation.py --model_name train_mts.pt --data data/wikitext-2/
