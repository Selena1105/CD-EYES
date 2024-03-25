import torch
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse

root_path = ".."
save_path = root_path + "/hub/results/TMP"
test_path = root_path + "/hub/datasets/photo-Train-Val-Test/test"

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args() 
    return args

def visualize(input_img_name, dm):
    try:
        input_img = plt.imread(test_path+'/'+input_img_name+'.jpg')[:,:,:3]
    except:
        input_img = plt.imread(test_path+'/'+input_img_name+'.jpg')
        input_img = np.array([input_img,input_img,input_img]).transpose((1,2,0))
    input_img = input_img/(np.max(input_img.flatten())) * 255.0

    dm = dm/(np.max(dm.flatten())) * 255.0
    dm = cv2.resize(dm, (input_img.shape[1],input_img.shape[0]))
    plt.imsave(save_path+'/'+input_img_name+'_Dm'+'.jpg', dm, cmap='hot')
    dm = plt.imread(save_path+'/'+input_img_name+'_Dm'+'.jpg')[:,:,:3]
    
    merge_img = np.array((input_img + dm)/2).astype('uint8')
    plt.imsave(save_path+'/'+input_img_name+'_Merge'+'.jpg', merge_img)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    idx = 0
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            model.return_hidden_features = False
            outputs = model(inputs)
            visualize(name[0], outputs.squeeze().detach().cpu().numpy())
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
            idx += 1

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)