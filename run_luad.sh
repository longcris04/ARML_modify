python 1_train_stage1.py --dataset luad --gpu 3  --trainroot /mnt/disk1/backup_user/22long.nh/ARML/datasets/LUAD-HistoSeg/train/ --batch_size 64 --testroot /mnt/disk1/backup_user/22long.nh/ARML/datasets/LUAD-HistoSeg/test/ --max_epoches 20 --weights ../ARML/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
python 2_generate_PM.py --dataroot /mnt/disk1/backup_user/22long.nh/ARML/datasets/LUAD-HistoSeg --dataset luad --gpu 3
python 3_train_stage2.py --dataset luad --dataroot /mnt/disk1/backup_user/22long.nh/ARML/datasets/LUAD-HistoSeg --epochs 10 --batch-size 32 --gpu-ids 3 --gpu 3

