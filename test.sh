python execute.py \
  --exp_name=baseline \
  --videos_dir=/home/leon/workspace/multimodal/dataset/MSR-VTT-CLIP4clip/MSRVTT_Videos \
  --mode=test \
  --arch=victor \
  --batch_size=64 \
  --loss=clip \
  --frozen_clip=0 \
  --device=cuda:1 \
  --noclip_lr=3e-4 \
  --transformer_dropout=0.3 \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k 










