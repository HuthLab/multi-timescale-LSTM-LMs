Code associated with ICLR 2021 paper: Mahto, S., Vo, V.A., Turek, J.S., Huth, A. "[Multi-timescale representation learning in LSTM language models.](https://openreview.net/forum?id=9ITXiTrAoT)"

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
python run_dyckn.py -u 256 -l 1 --epochs 2000 -s 200 --lr 1e-4 --batch_size 32 --seed 1 --model MTS --alpha 1.50 --scale 1.0 -o ./results/dyckn/MTS_u256_l1_e2000_b32_s200_lr0.0001_sc1.00_a1.50_seed1/
```

Use `--model MTS` for the multi-timescale LSTM model and `--model Baseline` for the baseline LSTM model.
The experiment in the paper used seeds {1..20} for both networks.


### Citation
Please, cite this paper as follows:

Mahto, S., Vo, V.A., Turek, J.S., Huth, A. "Multi-timescale representation learning in LSTM language models", International Conference on Learning Representations, May 2021.

```
@inproceedings{ mahto2021multitimescale,
    title={Multi-timescale Representation Learning in {\{}LSTM{\}} Language Models},
    author={Shivangi Mahto and Vy Ai Vo and Javier S. Turek and Alexander Huth},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=9ITXiTrAoT}
}
```
