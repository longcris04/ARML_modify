import os
import torch
import argparse
import importlib
import numpy as np
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from tool.gradcam import GradCam


def create_pseudo_mask_testval(model, dataroot, data_type, fm, savepath, n_class, palette, dataset, args):
    """
    Create pseudo masks for test/val sets using GradCAM.
    Uses model predictions instead of filename labels since test/val sets don't have labels in filenames.
    
    Args:
        model: The trained model
        dataroot: Root directory of the dataset
        data_type: 'test' or 'val'
        fm: Feature map to use ('b4_3', 'b4_5', 'b5_2', 'b6', 'bn7')
        savepath: Path to save the generated pseudo masks
        n_class: Number of classes
        palette: Color palette for visualization
        dataset: Dataset name ('luad' or 'bcss')
        args: Arguments object containing gpu, etc.
    """
    # Get device from args
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Select feature module
    if fm == 'b4_3':
        ffmm = model.b4_3
    elif fm == 'b4_5':
        ffmm = model.b4_5
    elif fm == 'b5_2':
        ffmm = model.b5_2
    elif fm == 'b6':
        ffmm = model.b6
    elif fm == 'bn7':
        ffmm = model.bn7
    else:
        print(f'Error: Invalid feature map {fm}')
        return
    
    print(f"Processing {data_type} set for dataset: {dataset}")
    
    # Setup data loader
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(
        data_path=os.path.join(dataroot, data_type, 'img'),
        transform=transform
    )
    
    infer_data_loader = DataLoader(
        infer_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        pin_memory=False
    )
    
    print(f"Generating pseudo masks for {data_type} set using feature map: {fm}")
    pbar = tqdm(infer_data_loader, desc=f'Creating PM ({data_type}/{fm})', ncols=120)
    
    for iter, (img_name, img_list) in enumerate(pbar):      
        img_name = img_name[0]
        img_path = os.path.join(dataroot, data_type, 'img', img_name + '.png')
        
        orig_img = np.asarray(Image.open(img_path))
        
        # Initialize GradCAM
        grad_cam = GradCam(
            model=model, 
            feature_module=ffmm,
            target_layer_names=["1"], 
            use_cuda=True, 
            device=device
        )
        
        # Generate CAM for each class
        cam = []
        for i in range(n_class):
            target_category = i
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        if _range > 0:
            norm_cam = (norm_cam - np.min(norm_cam)) / _range
        else:
            norm_cam = np.zeros_like(norm_cam)
        
        # Use model predictions to get labels instead of filename
        # This is the key difference from the training set approach
        with torch.no_grad():
            img_tensor = img_list.to(device)
            _, y = model.forward_cam(img_tensor)
            y = y.cpu().detach().numpy()[0]
            
            # Threshold to get binary labels (same as in infer function)
            label = torch.tensor([1.0 if j > 0.25 else 0.0 for j in y])
            
            # Ensure at least one class is present
            if torch.sum(label) == 0:
                label[np.argmax(y)] = 1.0
        
        # Generate segmentation map using predicted labels
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        
        # Handle background differently for different datasets
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        elif dataset == 'bcss':
            bg_score = np.zeros((1, orig_img.shape[0], orig_img.shape[1]))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score)
        
        # Save the pseudo mask
        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        save_filename = img_name.split('\\')[-1] + '.png'
        visualimg.save(os.path.join(savepath, save_filename), format='PNG')
        
        pbar.set_postfix({'Processed': iter + 1})
    
    pbar.close()
    print(f"Completed generating {iter + 1} pseudo masks for {data_type} set using {fm}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pseudo masks for test and validation sets')
    parser.add_argument("--gpu", default=3, type=int, help="GPU device number to use")
    parser.add_argument("--weights", default='/home/22long.nh/Projects/Pathology/WSSS/ARML_modify/checkpoints/stage1_checkpoint_trained_on_luad_res38d_arml.pth', 
                        type=str, help="Path to trained model weights")
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="/mnt/disk1/backup_user/22long.nh/ARML/datasets/LUAD-HistoSeg", type=str, 
                        help="Root directory of the dataset")
    parser.add_argument("--dataset", default="luad", type=str, choices=['bcss', 'luad'],
                        help="Dataset name")
    parser.add_argument("--data_types", default="test,val", type=str,
                        help="Comma-separated list of data types to process (e.g., 'test,val' or 'test')")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--feature_maps", default="b4_5,b5_2,bn7", type=str,
                        help="Comma-separated list of feature maps to use")

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
    
    # Setup palette based on dataset
    if args.dataset == 'luad':
        palette = [0] * 15
        palette[0:3] = [205, 51, 51]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [65, 105, 225]
        palette[9:12] = [255, 165, 0]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'bcss':
        palette = [0] * 15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [0, 0, 255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
    
    # Load model
    print(f"Loading model from {args.weights}")
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()
    model = model.to(device)
    
    # Parse data types and feature maps
    data_types = [dt.strip() for dt in args.data_types.split(',')]
    feature_maps = [fm.strip() for fm in args.feature_maps.split(',')]
    
    print(f"\nProcessing data types: {data_types}")
    print(f"Using feature maps: {feature_maps}")
    print("="*80)
    
    # Process each data type (test, val, etc.)
    for data_type in data_types:
        print(f"\n{'='*80}")
        print(f"Processing {data_type.upper()} set")
        print(f"{'='*80}")
        
        # Create base directory for this data type
        data_PM = f"{data_type}_PM"
        PMpath = os.path.join(args.dataroot, data_PM)
        
        if not os.path.exists(PMpath):
            os.makedirs(PMpath)
            print(f"Created directory: {PMpath}")
        
        # Process each feature map
        for fm in feature_maps:
            print(f"\n{'-'*80}")
            print(f"Feature Map: {fm}")
            print(f"{'-'*80}")
            
            savepath = os.path.join(PMpath, f'PM_res38d_arml{fm}')
            
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                print(f"Created directory: {savepath}")
            
            # Generate pseudo masks
            create_pseudo_mask_testval(
                model=model,
                dataroot=args.dataroot,
                data_type=data_type,
                fm=fm,
                savepath=savepath,
                n_class=args.n_class,
                palette=palette,
                dataset=args.dataset,
                args=args
            )
    
    print(f"\n{'='*80}")
    print("All pseudo masks generated successfully!")
    print(f"{'='*80}")
