import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=3, type=int, help="GPU device number to use")
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_bcss_res38d_arml.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="datasets/BCSS-WSSS", type=str)
    parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--data", default="train", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()
    
    # Force GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Setting CUDA_VISIBLE_DEVICES to GPU {args.gpu}")
    
    # After setting CUDA_VISIBLE_DEVICES, the visible GPU becomes device 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Update args.gpu to reflect the remapped device
    args.gpu = 0
    
    print(args)
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51]
        palette[3:6] = [0,255,0]
        palette[6:9] = [65,105,225]
        palette[9:12] = [255,165,0]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0,255,0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
    data_PM = args.data + "_PM"
    PMpath = os.path.join(args.dataroot,data_PM)

    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()
    model = model.to(device)
    #
    data = args.data
    #
    fm = 'b4_5'
    savepath = os.path.join(PMpath,'PM_'+'res38d_arml'+fm)
    print(savepath)
    # print(savepath_test)
    # print(savepath_val)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # if not os.path.exists(savepath_test):
        # os.mkdir(savepath_test)
    # if not os.path.exists(savepath_val):
        # os.mkdir(savepath_val)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args)
    ##
    fm = 'b5_2'
    savepath = os.path.join(PMpath,'PM_'+'res38d_arml'+fm)
    # savepath_test = os.path.join(PMpath_test,'PM_'+'res38d_arml'+fm)
    # savepath_val =  os.path.join(PMpath_val,'PM_'+'res38d_arml'+fm)
    print(savepath)
    # print(savepath_test)
    # print(savepath_val)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # if not os.path.exists(savepath_test):
    #     os.mkdir(savepath_test)
    # if not os.path.exists(savepath_val):
    #     os.mkdir(savepath_val)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args)
    #
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+'res38d_arml'+fm)
    # savepath_test = os.path.join(PMpath_test,'PM_'+'res38d_arml'+fm)
    # savepath_val =  os.path.join(PMpath_val,'PM_'+'res38d_arml'+fm)
    print(savepath)
    # print(savepath_test)
    # print(savepath_val)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # if not os.path.exists(savepath_test):
    #     os.mkdir(savepath_test)
    # if not os.path.exists(savepath_val):
    #     os.mkdir(savepath_val)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args)
