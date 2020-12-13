#!/bin/bash

# make sure to have installed and activated the correct packages


## BOW approaches

# BOW model
python3 -u train.py --model BOW --result_dir ./results/BOW/ --plot_dir ./plots/BOW/ --num_iterations 0 --eval_every 1000 --print_every 1000 --patience 20 --learning_rate 0.0003

# CBOW
python3 -u train.py --model CBOW --result_dir ./results/CBOW/ --plot_dir ./plots/CBOW/ --num_iterations 0 --eval_every 1000 --print_every 1000 --patience 20 --learning_rate 0.0003 --embed_dim 300

# Deep CBOW
python3 -u train.py --model DeepCBOW --result_dir ./results/DeepCBOW/ --plot_dir ./plots/DeepCBOW/ --num_iterations 0 --eval_every 1000 --print_every 1000 --patience 20 --learning_rate 0.0003 --embed_dim 300 --hidden_dim 100

# PT Deep CBOW
python3 -u train.py --model PTDeepCBOW --result_dir ./results/PTDeepCBOW/ --plot_dir ./plots/PTDeepCBOW/ --num_iterations 0 --eval_every 1000 --print_every 1000 --patience 20 --learning_rate 0.0003 --embed_dim 300 --hidden_dim 100 --use_pt_embed True


## LSTM approaches

# LSTM
python3 -u train.py --model LSTM --result_dir ./results/LSTM/ --plot_dir ./plots/LSTM/ --use_pt_embed True --learning_rate 0.0003 --num_iterations 0 --print_every 50 --eval_every 50 --patience 20 --hidden_dim 150

# TreeLSTM
python3 -u train.py --model TreeLSTM --result_dir ./results/TreeLSTM/ --plot_dir ./plots/TreeLSTM/ --use_pt_embed True --learning_rate 0.0003 --num_iterations 0 --print_every 50 --eval_every 50 --patience 20 --hidden_dim 150

# word order (LSTM)
python3 -u train.py --result_dir ./results/word_order/ --permute True --plot_dir ./plots/word_order/ --use_pt_embed True --learning_rate 0.0003 --hidden_dim 150 --num_iterations 0 --patience 20 --print_every 50 --eval_every 20

# subtree generation (TreeLSTM)
python3 -u train.py --model TreeLSTM --result_dir ./results/subtrees/ --plot_dir ./plots/subtrees/ --use_pt_embed True --learning_rate 0.0003 --print_every 50 --eval_every 50 --num_iterations 0 --hidden_dim 150 --create_subtrees True --patience 20

# childsum TreeLSTM
python3 -u train.py --model TreeLSTM --result_dir ./results/childsum/ --plot_dir ./plots/childsum/ --use_pt_embed True --learning_rate 0.0003 --num_iterations 0 --print_every 50 --eval_every 50 --patience 20 --hidden_dim 150 --childsum True



