import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np

# # Comprehensive fix for NumPy 2.0 compatibility with MXNet
# # Add back deprecated constants
# if not hasattr(np, 'PZERO'):
#     np.PZERO = 0.0
# if not hasattr(np, 'NZERO'):
#     np.NZERO = -0.0
# if not hasattr(np, 'PINF'):
#     np.PINF = np.inf
# if not hasattr(np, 'NINF'):
#     np.NINF = -np.inf
# if not hasattr(np, 'NAN'):
#     np.NAN = np.nan
# if not hasattr(np, 'NaN'):
#     np.NaN = np.nan

# # Add back deprecated function aliases
# if not hasattr(np, 'alltrue'):
#     np.alltrue = np.all

# # Add back financial functions that were moved to numpy-financial
# if not hasattr(np, 'mirr'):
#     try:
#         import numpy_financial as npf
#         np.mirr = npf.mirr
#     except ImportError:
#         # Provide a dummy function if numpy-financial is not installed
#         def mirr(*args, **kwargs):
#             raise NotImplementedError("numpy.mirr was removed. Install numpy-financial package.")
#         np.mirr = mirr

import argparse
import importlib
import random
import pandas as pd
from tqdm import tqdm
from visdom import Visdom
import network.resnet38_cls
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
# cudnn.enabled = True

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def train_phase(args):
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # viz = Visdom(env=args.env_name)
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.GaussianBlur(3),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                          transforms.ToTensor()])
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    avg_meter = pyutils.AverageMeter(
            'loss',
            'avg_ep_EM',
            'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    
    for ep in range(args.max_epoches):
        model.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0
        
        # Progress bar for each epoch
        pbar = tqdm(train_data_loader, 
                   desc=f"Epoch {ep+1}/{args.max_epoches}",
                   total=len(train_data_loader),
                   ncols=120)
        
        for iter, (filename, data, label) in enumerate(pbar):
            img = data
            # label = label.to(device, non_blocking=True)
            # label = label.to(device)
            if ep > 1:
                # enable_PDA = 1
                enable_AMM = 1
                # enable_NAEA = 1
                enable_MARS = 1
            else:
                # enable_PDA = 0
                enable_AMM = 0
                # enable_NAEA = 0
                enable_MARS = 0
            
            # Ensure all tensors are moved to the correct device
            # img = img.to(device, non_blocking=True)  # Move input data to device
            # label = label.to(device, non_blocking=True)  # Move labels to device
            img = img.to(device)
            label = label.to(device)

            # Debugging: Verify tensor devices
            assert img.device == device, f"Input data is on {img.device}, expected {device}"
            assert label.device == device, f"Labels are on {label.device}, expected {device}"

            # Model forward pass
            x, feature, y, cam1 = model(img, enable_PDA=0, enable_AMM=0, enable_NAEA=0, enable_MARS=enable_MARS)
            prob = y.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls) == True:  # exact match
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc
            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)
            loss_cls = F.multilabel_soft_margin_loss(x, label, reduction='none')
            loss = loss_cls.mean()
            # print(loss)
            avg_meter.add({'loss':loss.item(),
                            'avg_ep_EM':avg_ep_EM,
                            'avg_ep_acc':avg_ep_acc,
                           })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{avg_meter.get("loss"):.4f}',
                'EM': f'{avg_ep_EM:.4f}',
                'Acc': f'{avg_ep_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            if (optimizer.global_step)%100 == 0 and (optimizer.global_step)!=0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'avg_ep_EM:%.4f' % (avg_meter.get('avg_ep_EM')),
                      'avg_ep_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
            
            if iter == 3:
                break
        
        pbar.close()
        if model.gama > 0.65:
            model.gama = model.gama*0.98
        print('Gama of progressive dropout attention is: ',model.gama)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        save_name = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth')
        torch.save(model.state_dict(), save_name)

def test_phase(args):
    # For multi-GPU inference, model needs to be on cuda:0
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for testing: {device}")
    print(f"Note: Model will be replicated across {torch.cuda.device_count()} GPUs for inference")
    # exit(0)
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.to(device)

    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth')
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    print(f"finished loading model weights from {args.weights}")
    # Create result directory if it doesn't exist
    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)
    
    csv_path = os.path.join(args.csv_dir, f'stage1_test_results_on_{args.dataset}.csv')
    
    # Read existing CSV or create new one
    if os.path.exists(csv_path):
        result_df = pd.read_csv(csv_path, index_col=0)
    else:
        result_df = pd.DataFrame(columns=['iou_0', 'iou_1', 'iou_2', 'iou_3', 'miou',
                                          'dice_0', 'dice_1', 'dice_2', 'dice_3', 'mdice'])
    
    # Run inference
    score = infer(model, args.testroot, args.n_class, args)
    print(score)
    
    # Extract metrics
    cls_iou = score["Class IoU"]
    cls_dice = score.get("Dice Coefficients", {})
    
    # If Dice Coefficients not in score, calculate from IoU
    if not cls_dice:
        cls_dice = {i: 2*cls_iou[i]/(1+cls_iou[i]) if cls_iou[i] > 0 else 0.0 
                   for i in range(args.n_class)}
    
    mIoU = score["Mean IoU"]
    mDice = score.get("Mean Dice", score.get("Dice", 0.0))
    
    # Prepare new row
    iou_0, iou_1, iou_2, iou_3 = cls_iou[0], cls_iou[1], cls_iou[2], cls_iou[3]
    dice_0, dice_1, dice_2, dice_3 = cls_dice[0], cls_dice[1], cls_dice[2], cls_dice[3]
    
    new_row = pd.DataFrame({
        'iou_0': [iou_0], 'iou_1': [iou_1], 'iou_2': [iou_2], 'iou_3': [iou_3],
        'miou': [mIoU],
        'dice_0': [dice_0], 'dice_1': [dice_1], 'dice_2': [dice_2], 'dice_3': [dice_3],
        'mdice': [mDice]
    })
    
    # Append to dataframe
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    
    # Save to CSV
    result_df.to_csv(csv_path, index=True)
    print(f"\nResults saved to: {csv_path}")

    # torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth'))
    # torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=3, type=int, help="GPU device number to use")
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="Stage 1", type=str)
    parser.add_argument("--env_name", default="PDA", type=str)
    parser.add_argument("--model_name", default='res38d_arml', type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/BCSS-WSSS/train/', type=str)
    parser.add_argument("--testroot", default='datasets/BCSS-WSSS/test/', type=str)
    parser.add_argument("--save_folder", default='checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='bcss', type=str)
    parser.add_argument("--csv_dir", default='./result', type=str)
    args = parser.parse_args()
    # gpu = 0
    # device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())
    # torch.cuda.empty_cache()

    # train_phase(args)
    test_phase(args)
