import h5py
import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm



def png2hdf(png_dir, hdf_dir):
    # png_dir = '/home/zheng/Desktop/wafer4/png'
    # hdf_dir = '/home/zheng/Desktop/wafer4/hdf'
    if not os.path.exists(hdf_dir):
        os.mkdir(hdf_dir)
    png_list = sorted(glob(os.path.join(png_dir, '*.png')))
    num = len(png_list)
    for k,png_path in tqdm(enumerate(png_list)):
        png_name = os.path.basename(png_path)
        hdf_path = os.path.join(hdf_dir, png_name.replace('.png', '.h5'))
        img = Image.open(png_path)
        if k==0:
            h,w = img.size
            save_shape = (num,h,w)
            save_hdf = np.zeros(save_shape, dtype=np.uint8)
        img = np.array(img)
        save_hdf[k] = img
    with h5py.File(os.path.join(hdf_dir, 'wafer.h5'), 'w') as f:
        f.create_dataset('main', data=save_hdf, dtype=np.uint8)
    print('png2hdf done.')

def read_hdf(hdf_path):
    with h5py.File(hdf_path, 'r') as f:
        data = f['main'][:]
    return data

if __name__ == '__main__':
    png_dir = '/braindat/lab/chenyd/DATASET/EM_data_ours/wafer26_raw'
    hdf_dir = '/braindat/lab/liuxy/Backbones/data/wafer26'
    png2hdf(png_dir, hdf_dir)