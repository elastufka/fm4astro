import torch
import argparse
import sys
import glob
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default="features.pth", type=str,help='')
    args = parser.parse_args()
    return args

def run_main(args):
    aa = torch.load(args.file_name).cpu()
    print(aa.shape)
    if aa.ndim ==4 :
        bb = aa.mean(axis=-1).mean(axis=-1)
        print(bb.shape)
        output_filename = f"{args.file_name[:-4]}_mean.pth"
        torch.save(bb, output_filename)
    elif aa.ndim > 2:
        bb = aa[:,0,:].numpy()

        if "MGCLS" in args.file_name and "test" in args.file_name:
            bb = np.delete(bb, [125, 129, 153], axis= 0)
        if "MGCLS" in args.file_name and "train" in args.file_name:
            bb = np.delete(bb, [427,440,441,455,462,463,469,470,479,483,484,504,506,514,528,536,543,551,552,567,1817], axis= 0)
        print(bb.shape)
        output_filename = f"{args.file_name[:-4]}_CLS.npy"
        np.save(output_filename,bb)

if __name__ == "__main__":
    args = get_args()
    run_main(args)