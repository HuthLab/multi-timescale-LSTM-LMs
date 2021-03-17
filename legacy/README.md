Here we give instructions to train models that will more closely reproduce the language modeling results reported in Merity et al. 2017, "Regularizing and optimizing LSTM language models". The multi-timescale LSTM (MTS-LSTM) can also be trained in this legacy mode.

## Required dependencies: Python3.6 or above, Numpy, Scipy and Pytorch-0.4

### To train a multi-timescale model:

#### On PTB data

python train_legacy.py --batch_size 20 --data ../data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1000 --save train_mts.pt 

#### On Wiki data 

python train_legacy.py --data ../data/wikitext-2 --dropouth 0.2 --seed 1882 --epoch 1000 --save train_mts.pt 


### To train the original AWD-LSTM model:

#### On PTB data

python train_legacy.py --batch_size 20 --data ../data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save train_baseline_ptb.pt --baseline

