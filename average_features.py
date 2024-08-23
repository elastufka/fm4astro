import torch
import argparse
import sys
import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default="features.pth", type=str,help='')
    args = parser.parse_args()
    return args

def run_main(args):
    aa = torch.load(args.file_name).cpu()
    if aa.ndim > 2:
        bb = aa.mean(axis=1)
        output_filename = f"{args.file_name[:-4]}_mean.pth"
        torch.save(bb, output_filename)

if __name__ == "__main__":
    args = get_args()
    run_main(args)