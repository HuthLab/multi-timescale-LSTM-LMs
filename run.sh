#!/bin/bash

ptb_specs="--batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141"
echo Training baseline and multi-timescale LM on PTB dataset.
#get ptb data downloaded on local machine , in case you want Wiki data , please read the details in getdata.sh
bash getdata.sh
#train a standard/baseline model here
python train_mts.py $ptb_specs  --epoch 1000 --save baseline_ptb.pt --baseline > results_baseline.txt
#train a standard/baseline model here
python train_mts.py $ptb_specs  --epoch 1000 --save mts_ptb.pt > result_mts.txt
#evaluate both the models on ptb test data
echo Evaluate Multi-timescale LSTM LM
python util/model_evaluation.py --model_name baseline_ptb.pt --data data/penn/ --model_performance
echo Evaluate Standard LSTM LM
python util/model_evaluation.py --model_name mts_ptb.pt --data data/penn/ --model_performance
echo Perform statistical evaluation
python util/model_evaluation.py --model_name ~/new_awd/awd-lstm-lm/mts_ptb.pt --baseline_model ~/new_awd/awd-lstm-lm/baseline_ptb.pt --significance_testing --data data/penn --cuda
