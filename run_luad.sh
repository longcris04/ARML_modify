python 1_train_stage1.py --dataset luad --gpu 1  --trainroot datasets/LUAD-HistoSeg/train/ --batch_size 32 --testroot dataset/LUAD-HistoSeg/test/ --max_epoches 20
python 2_generate_PM.py --dataroot datasets/LUAD-HistoSeg --dataset luad --gpu 1
python 3_train_stage2.py --dataset luad --dataroot datasets/LUAD-HistoSeg --epochs 10 --batch-size 64 --gpu-ids 4 --gpu 4

