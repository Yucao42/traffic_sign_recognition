MODEL='resnext_flow'

SAVE_DIR=Saves/expriments/${MODEL}

python3 \
train.py \
--basemodel ${MODEL} \
--save ${SAVE_DIR} \
--epochs 200 \
--minibatch_size 512 \
--valid_bs 288 \
--valid_nworker 8 \
--learning_rate 1e-2 \
--momentum 0.9 \
--decay 5e-4 \
--schedule 60 80 100 \
--gamma 0.1 \
--ngpu 8 \
--nworker 32 \
--train_clip 1 \
--start_epoch 0 \
--benchmark open \
--valid_clip 1 \
--neg_pos_ratio 1 \
--n_classes 60 \
--input_size 224 224 \
--base_stop 100 \
--print_freq 10 \
--eval_freq 1 \
--save_freq 5 \
--schedule 60 80 100 \
--base_stop 100 \
--aug 0 \
#--load ${SAVE_DIR}/../resnet50_big/model_best.pytorch \
