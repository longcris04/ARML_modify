import argparse
import os
import numpy as np
import math
import random
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
# torch.autograd.set_detect_anomaly(True)
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from network.pspnet import PSPNet
from tool.loss import SegmentationLosses, STLoss
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
import ml_collections
import segmentation_models_pytorch as smp
from collections import defaultdict
from scipy import stats
from torchvision import transforms
import timm

# torch.backends.cudnn.benchmark = False

def SWV(outputs_main, outputs_aux1, outputs_aux2, mask):
    n = outputs_main.shape[0]
    loss_main = F.cross_entropy(
        outputs_main, mask.long(), reduction='none').view(n, -1)
    hard_aux1 = torch.argmax(outputs_aux1, dim=1).view(n, -1)
    hard_aux2 = torch.argmax(outputs_aux2, dim=1).view(n, -1)
    loss_select = 0
    for i in range(n):
        aux1_sample = hard_aux1[i]
        aux2_sample = hard_aux2[i]
        loss_sample = loss_main[i]
        agree_aux = (aux1_sample == aux2_sample)
        disagree_aux = (aux1_sample != aux2_sample)
        loss_select += 2*torch.sum(loss_sample[agree_aux]) + (1/2)*torch.sum(loss_sample[disagree_aux])

    return loss_select / (n*loss_main.shape[1])

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.gama = 1.0
        # Define
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        # model = DeepLab(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn)
        model = smp.PSPNet(encoder_name='timm-resnest101e', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        lr = args.lr
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Segmentation Model - Total parameters: {total_params:,}")
        print(f"Segmentation Model - Trainable parameters: {trainable_params:,}")
        
        # Setup device for single GPU training
        self.device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
        print(f"Using device: {self.device}")
        if args.cuda:
            self.model = self.model.to(self.device)
        # Resuming checkpoint
        self.best_pred = 0.0
        self.start_epoch = 0
        self.save_path = args.savepath
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Load checkpoint for resuming training
        if args.resume is not None:
            checkpoint_path = os.path.join('checkpoints', 
                f'stage2_checkpoint_trained_on_{args.dataset}{args.backbone}{args.loss_type}.pth')
            
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(f"=> no checkpoint found at '{checkpoint_path}'")
            
            print(f"=> Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            state_dict = checkpoint['state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            
            # Load optimizer state if not fine-tuning
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load epoch and best_pred if available
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"=> Resuming from epoch {self.start_epoch}")
            
            if 'best_pred' in checkpoint:
                self.best_pred = checkpoint['best_pred']
                print(f"=> Previous best mIoU: {self.best_pred:.4f}")
            
            print(f"=> Successfully loaded checkpoint '{checkpoint_path}'")
        







        # if args.resume is not None:
            # if not os.path.isfile(args.resume):
            #     raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            
            # checkpoint = torch.load(args.resume, map_location=self.device)
            # # Load state dict - handle both DataParallel and non-DataParallel checkpoints
            # state_dict = checkpoint['state_dict']
            # # Remove 'module.' prefix if present (from DataParallel)
            # if list(state_dict.keys())[0].startswith('module.'):
            #     state_dict = {k[7:]: v for k, v in state_dict.items()}
            # self.model.load_state_dict(state_dict, strict=False)
            # # if args.ft:
            # #     self.optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' ".format(args.resume))

        

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target, target_a, target_b = sample['image'], sample['label'], sample['label_a'], sample['label_b']
            if self.args.cuda:
                image = image.to(self.device)
                target = target.to(self.device)
                target_a = target_a.to(self.device)
                target_b = target_b.to(self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            output2 = self.model(image)
            output3 = self.model(image)
            one = torch.ones((output.shape[0],1,224,224), device=self.device)
            one2 = torch.ones((output2.shape[0],1,224,224), device=self.device)
            one3 = torch.ones((output3.shape[0],1,224,224), device=self.device)
            output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            output2 = torch.cat([output2,(100 * one2 * (target==4).unsqueeze(dim = 1))],dim = 1)
            output3 = torch.cat([output3,(100 * one3 * (target==4).unsqueeze(dim = 1))],dim = 1)
            loss_o = self.criterion(output, target, self.gama)
            # loss_a = self.criterion(output, target_a, self.gama)
            # loss_b = self.criterion(output, target_b, self.gama)
            loss_v1 = SWV(output, output2, output3, target)
            loss_st1 = STLoss()(output, output2)
            loss_st2 = STLoss()(output, output3)
            loss_st = (loss_st1 + loss_st2) / 2
            loss = 0.8*loss_v1 + 0.2*loss_st
            # loss = 0.6 * loss_o + 0.2 * loss_a + 0.2 * loss_b
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        # if (epoch + 1) % 3 == 0:
        self.gama = self.gama * 0.98

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image = image.to(self.device)
                target = target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        DSC = self.evaluator.Dice_Similarity_Coefficient()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/DSC', DSC, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, DSC: {}".format(Acc, Acc_class, mIoU, FWIoU, DSC))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred
            }, 'stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')

    def load_the_best_checkpoint(self):
        checkpoint_path = os.path.join('checkpoints', 
            f'stage2_checkpoint_trained_on_{self.args.dataset}{self.args.backbone}{self.args.loss_type}.pth')
        
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at: {checkpoint_path}\n"
                f"Please ensure you have trained the model first or provide the correct checkpoint path."
            )
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle both DataParallel and non-DataParallel checkpoints
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")
        
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            image_name = sample[-1][0].split('/')[-1].replace('.png', '')
            if self.args.cuda:
                image = image.to(self.device)
                target = target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    # print(output.shape)
                    _,y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    # y_cls = y_cls.cpu().data
                    # print(y_cls)
                    pred_cls = (y_cls > 0.5)
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        DSC = self.evaluator.Dice_Similarity_Coefficient()
        dices = self.evaluator.Dice_Coefficient_Per_Class()
        
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/DSC', DSC, epoch)
        print('Test:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, DSC: {}".format(Acc, Acc_class, mIoU, FWIoU, DSC))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)
        print('Dices: ', dices)
        
        # Save results to CSV
        csv_dir = getattr(self.args, 'csv_dir', './result')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        csv_path = os.path.join(csv_dir, f'stage2_test_results_on_{self.args.dataset}.csv')
        
        # Read existing CSV or create new one
        if os.path.exists(csv_path):
            result_df = pd.read_csv(csv_path, index_col=0)
        else:
            result_df = pd.DataFrame(columns=['iou_0', 'iou_1', 'iou_2', 'iou_3', 'miou',
                                              'dice_0', 'dice_1', 'dice_2', 'dice_3', 'mdice'])
        
        # Prepare new row - get first 4 classes (excluding background/ignore class)
        iou_0 = ious[0] 
        iou_1 = ious[1] 
        iou_2 = ious[2] 
        iou_3 = ious[3] 
        
        dice_0 = dices[0]
        dice_1 = dices[1] 
        dice_2 = dices[2] 
        dice_3 = dices[3] 
        
        new_row = pd.DataFrame({
            'iou_0': [iou_0], 'iou_1': [iou_1], 'iou_2': [iou_2], 'iou_3': [iou_3],
            'miou': [mIoU],
            'dice_0': [dice_0], 'dice_1': [dice_1], 'dice_2': [dice_2], 'dice_3': [dice_3],
            'mdice': [DSC]
        })
        
        # Append to dataframe
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        
        # Save to CSV
        result_df.to_csv(csv_path, index=True)
        print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2")
    parser.add_argument('--gpu', type=int, default=3, help='GPU device number to use')
    parser.add_argument('--backbone', type=str, default='psp101_cca_ar_mix_distonly_', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--Is_GM', type=bool, default=False, help='Enable the Gate mechanism in test phase')
    parser.add_argument('--dataroot', type=str, default='datasets/BCSS-WSSS/')
    parser.add_argument('--dataset', type=str, default='bcss')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=0, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='mvce', choices=['ce', 'mvce'])
    parser.add_argument('--n_class', type=int, default=4)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N')
    # optimizer params
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint or "True" to auto-resume')
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False, help='Fine-tune mode: do not load optimizer state')
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--csv_dir', type=str, default='./result', help='Directory to save CSV results')
    parser.add_argument('--continue-training', action='store_true', default=False, help='Continue training from last checkpoint')
    args = parser.parse_args()
    
    # Auto-set resume if continue-training is enabled
    if args.continue_training:
        args.resume = 'True'
    
    # Force GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Setting CUDA_VISIBLE_DEVICES to GPU {args.gpu}")
    
    # After setting CUDA_VISIBLE_DEVICES, the visible GPU becomes device 0
    args.gpu = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Single GPU training - no need for sync_bn
    args.sync_bn = False
    print(args)
    trainer = Trainer(args)
    
    # Training loop - will start from start_epoch if resuming
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == 0:
            trainer.validation(epoch)
    
    # Run final test with best checkpoint
    print("\n" + "="*50)
    print("Training completed. Running final test...")
    print("="*50)
    trainer.test(args.epochs - 1, args.Is_GM)
    trainer.writer.close()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
