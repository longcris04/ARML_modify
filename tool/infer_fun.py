import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam, SmoothScoreCAM
from tqdm import tqdm
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img
def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
    return Combine

def infer(model, testroot,valroot, n_class, args):
    
    
    save_mask_path_test = os.path.join(testroot,'pred','stage1')
    if not os.path.exists(save_mask_path_test):
        os.makedirs(save_mask_path_test)
        
    # save_mask_path_val = os.path.join(valroot,'pred','stage1')
    # if not os.path.exists(save_mask_path_val):
    #     os.makedirs(save_mask_path_val)

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

    

    model.eval()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # print(f"in infer")
    # Ensure model is on the correct device
    model = model.to(device)
    # print(f"using device: {device}")
    
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(testroot,'img'),transform=transform)
    # print(f"len infer dataset: {len(infer_dataset)}")
    # exit(0)
    infer_data_loader = DataLoader(infer_dataset,
                                # batch_size=args.batch_size,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        # print(img_name)
        # print(f"here")
        # print(f"image name: {img_name}")
        
        # print(f"img_list shape: {img_list.shape}")
        
        # exit(0)
        img_name = img_name[0].split('\\')[-1]
        # print(img_name)

        img_path = os.path.join(os.path.join(testroot,'img'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.25):
            with torch.no_grad():
                cam, y = model.forward_cam(img.to(device))
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                # append at least one class
                if torch.sum(label) == 0:
                    label[np.argmax(torch.tensor(y))] = 1.0
                    
                cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()

                # print(f"CAM shape, dtype, min, mean, max: {cam.shape}, {cam.dtype}, {np.min(cam)}, {np.mean(cam)}, {np.max(cam)}")
                # print(f"Label shape, dtype, min, mean, max: {label.shape}, {label.dtype}, {torch.min(label)}, {torch.mean(label)}, {torch.max(label)}")
                # print(f"EXIT HERE!")
                # exit(0)
                return cam, label

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam) + 1e-5)

        # cam --> segmap
        if np.where(label==1)[0].size != 0:
            cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
            cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
            seg_map = infer_utils.cam_npy_to_label_map(cam_score)
            if iter%100==0:
                print(iter)
                # print(img_name)
            # print(f"Generated seg_map shape, dtype, min, max, unique: {seg_map.shape}, {seg_map.dtype}, {np.min(seg_map)}, {np.max(seg_map)}, {np.unique(seg_map)}")
            # print(f"EXIT HERE!")
            # exit(0)

            # save seg_map using Image palette to save path: 
            seg_map_save_path = os.path.join(save_mask_path_test, img_name + '.png')
            visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
            visualimg.putpalette(palette)
            visualimg.save(seg_map_save_path, format='PNG')

            cam_list.append(seg_map)
            gt_map_path = os.path.join(os.path.join(testroot,'mask'), img_name + '.png')
            # gt_map_path = os.path.join(os.path.join(dataroot,'mask'), '.png')
            gt_map = np.array(Image.open(gt_map_path))
            gt_list.append(gt_map)

    print(f"gt_list length: {len(gt_list)}, cam_list length: {len(cam_list)}")

    return iouutils.scores(gt_list, cam_list, n_class=n_class)

      
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset, args):
    # Get device from args
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # print(model)
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(dataset)
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'train'),transform=transform)
    
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                # batch_size=args.batch_size,
                                batch_size=1,
                                num_workers=8,
                                pin_memory=False)
    
    print(f"Generating pseudo masks using feature map: {fm}")
    pbar = tqdm(infer_data_loader, desc=f'Creating PM ({fm})', ncols=120)
    for iter, (img_name, img_list) in enumerate(pbar):      
        img_name = img_name[0]
        # print(img_name)
        # img_path = os.path.join(os.path.join(dataroot,''),img_name.split('\\')[1] + '/' + img_name.split('\\')[2] + '.png')
        img_path = os.path.join(os.path.join(dataroot,'train','img'),img_name+'.png')

        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCam(model=model, feature_module=ffmm, \
                target_layer_names=["1"], use_cuda=True, device=device)
        # print(model.fc8)
        # sscam = SmoothScoreCAM(model=model, target_layer=model.fc8)
        # sscam.register_hooks()
        # sscam = np.array(sscam)
        # print(grad_cam)
        # print(sscam)
        cam = []
        for i in range(n_class):
            target_category = i
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        norm_cam = (norm_cam - np.min(norm_cam))/_range
        # print(norm_cam)

        # cam2 = []
        # for i in range(n_class):
        #     target_category = i
        #     sscams = sscam.generate(img_list, target_category)
        #     cam2.append(sscams)
        # norm_cam = np.array(cam2)
        # _range = np.max(norm_cam) - np.min(norm_cam)
        # norm_cam = (norm_cam - np.min(norm_cam))/_range
        # print(norm_cam)
        ##  Extract the image-level label from the filename
        ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
        ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
        label_str = img_name.split(']')[0].split('[')[-1]
        if dataset == 'luad':
            label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
        elif dataset == 'bcss':
            label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
        ##  "bg_score" is the white area generated by "cv2.threshold".
        ##  Since lungs are the main organ of the respiratory system. There are a lot of alveoli (some air sacs) serving for exchanging the oxygen and carbon dioxide, which forms some white background in WSIs.
        ##  For LUAD-HistoSeg, we uses it in the pseudo-annotation generation phase to avoid some meaningless areas to participate in the training phase of stage2.
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        ##  Since the white background of images of breast cancer is meaningful (e.g. fat, etc), we do not use it for the training set of BCSS-WSSS.
        elif dataset == 'bcss':
            bg_score = np.zeros((1,224,224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score)
        # print(seg_map)
        visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, img_name.split('\\')[-1] +'.png'), format='PNG')
        
        pbar.set_postfix({'Processed': iter+1})
    
    pbar.close()
    print(f"Completed generating {iter+1} pseudo masks for {fm}")
