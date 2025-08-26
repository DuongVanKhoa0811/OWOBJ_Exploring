#!/usr/bin/env bash

echo running training of OWOBJ, M-OWODB dataset

set -x

EXP_DIR=exps/MOWODB/OWOBJ
PY_ARGS=${@:1}
WANDB_NAME=PROB_t1

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19\
    --train_set 't1_train' --test_set 'test' --epochs 41 --cls_loss_coef 2 --focal_alpha 0.25 \
    --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 \
    --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t1.txt
    

# PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset OWDETR \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set 't2_train' --test_set 'test' --epochs 51\
    --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 --freeze_prob_model \
    --exemplar_replay_max_length 1743 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t1_ft.txt" --exemplar_replay_cur_file "learned_owod_t2_ft.txt"\
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5\--lr 2e-5\
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t2.txt
    

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset OWDETR \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set "${WANDB_NAME}/learned_owod_t2_ft" --test_set 'test'  --epochs 111 --lr_drop 40\
    --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 \
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth"\
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t2_ft.txt
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset OWDETR \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20  --train_set 't3_train' --test_set 'test' --epochs 121\
    --model_type 'sketch' --obj_loss_coef 8e-4 --freeze_prob_model --obj_temp 1.3 --obj_kl_div 0.1 \
    --exemplar_replay_max_length 2361 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t2_ft.txt" --exemplar_replay_cur_file "learned_owod_t3_ft.txt"\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --lr 2e-5 \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t3.txt
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset OWDETR \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set "${WANDB_NAME}/learned_owod_t3_ft" --test_set 'test' --epochs 181 --lr_drop 35\
    --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 \
    --pretrain "${EXP_DIR}/t3/checkpoint0120.pth"\
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t3_ft.txt
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset OWDETR \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_train' --test_set 'test'  --epochs 191 \
    --model_type 'sketch' --obj_loss_coef 8e-4 --freeze_prob_model --obj_temp 1.3 --obj_kl_div 0.1 \
    --exemplar_replay_max_length 2749 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t3_ft.txt" --exemplar_replay_cur_file "learned_owod_t4_ft.txt"\
    --num_inst_per_class 40\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --lr 2e-5\
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t4.txt
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set "${WANDB_NAME}/learned_owod_t4_ft" --test_set 'test' --epochs 261 --lr_drop 50\
    --model_type 'sketch' --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 \
    --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/logs/t4_ft.txt