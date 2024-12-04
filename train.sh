python execute.py \
  --exp_name=baseline \
  --videos_dir=/home/leon/workspace/multimodal/dataset/MSR-VTT-CLIP4clip/MSRVTT_Videos \
  --arch=victor \
  --mode=train \
  --batch_size=64 \
  --loss=clip \
  --device=cuda:1 \
  --seed=8 \
  --noclip_lr=3e-5 \
  --transformer_dropout=0.3 \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k \
  --evals_per_epoch=5 \
  --num_epochs=5







