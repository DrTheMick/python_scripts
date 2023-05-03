#!/mnt/medic/mmartynowycz/miniconda3/bin/python

import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

def read_input_file(input_file):
    with open(input_file, 'rb') as f:
        f.read(512)
        data = np.fromfile(f, dtype=np.int16)
    
    dim = int(np.sqrt(data.size))
    data = data.reshape((dim, dim))
    data = torch.from_numpy(data).unsqueeze(0)
    
    return data

def bin_data_xyz(data, bin_x, bin_y, bin_z):
    z_dim, x_dim, y_dim = data.shape
    new_dim_x = x_dim // bin_x
    new_dim_y = y_dim // bin_y
    new_dim_z = z_dim // bin_z
    
    binned_data_xy = data.view(z_dim, new_dim_x, bin_x, new_dim_y, bin_y).sum(4).sum(2)
    
    binned_data_xyz = torch.zeros((new_dim_z, new_dim_x, new_dim_y), dtype=torch.uint16)
    
    for z in range(0, z_dim, bin_z):
        binned_data_xyz[z // bin_z] = binned_data_xy[z:z + bin_z].sum(dim=0)
    
    return binned_data_xyz.cpu().numpy()

#def bin_data_xy(data, bin_x, bin_y):
#    z_dim, x_dim, y_dim = data.shape[0], data.shape[1], data.shape[2]
#    new_x_dim = x_dim // bin_x
#    new_y_dim = y_dim // bin_y
#    binned_data = torch.zeros((z_dim, new_x_dim, new_y_dim), dtype=torch.int16, device='cuda')
#
#    for i in range(new_x_dim):
#        for j in range(new_y_dim):
#            binned_data[:, i, j] = torch.sum(data[:, i*bin_x:(i+1)*bin_x, j*bin_y:(j+1)*bin_y], dim=(1,2))#
#
#    return binned_data#
#
#def bin_data_z(data, bin_z):
#    z_dim, x_dim, y_dim = data.shape
#    new_z_dim = z_dim // bin_z
#    binned_data = torch.zeros((new_z_dim, x_dim, y_dim), dtype=torch.int16, device='cuda')

#    for i in range(new_z_dim):
#        binned_data[i, :, :] = torch.sum(data[i*bin_z:(i+1)*bin_z, :, :], dim=0)

#    return binned_data

def main():
    parser = argparse.ArgumentParser(description='Binning images in X, Y, and Z dimensions')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('bin_x', type=int, help='Binning factor in X dimension')
    parser.add_argument('bin_y', type=int, help='Binning factor in Y dimension')
    parser.add_argument('bin_z', type=int, help='Binning factor in Z dimension')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.img')]
    input_files.sort()

    print(f"Found {len(input_files)} input files in {args.input_dir}")
    print(f"Output files will be saved in {args.output_dir}")
    print(f"Images will be binned by {args.bin_x} in X, {args.bin_y} in Y, and {args.bin_z} in Z")

    data = [read_input_file(f) for f in input_files]
    data = torch.stack(data, dim=0).to('cuda')

    #binned_data_xy = bin_data_xy(data, args.bin_x, args.bin_y)
    #binned_data_xyz = bin_data_z(binned_data_xy, args.bin_z)
    binned_data_xyz = bin_data_xyz(data, args.bin_x, args.bin_y, args.bin_z)

    for idx, img in enumerate(tqdm(binned_data_xyz)):
        output_file = os.path.join(args.output_dir, f"binned_{args.bin_x}_{args.bin_y}_{args.bin_z}_{str(idx).zfill(4)}.tif")
        img = img.to('cpu').numpy().astype(np.uint16)
        print(f"Image shape: {img.shape}, data type: {img.dtype}")
        cv2.imwrite(output_file, img)

    print(f"Generated {len(os.listdir(args.output_dir))} output files in {args.output_dir}")

if __name__ == '__main__':
    main()
