import torch
from transformers import pipeline
from torchvision.transforms.functional import to_pil_image
import argparse
import sys
from fm_datasets import *
from galaxy_mnist.galaxy_mnist import GalaxyMNIST

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="facebook/dinov2-base", type=str,help='')
    parser.add_argument('--output_dir', default="/home/users/l/lastufka/FM_compare", type=str,help='')
    parser.add_argument('--freeze_at', default=2, type=int,help='')
    parser.add_argument('--ims_per_batch', default=8, type=int,help='')
    parser.add_argument('--imsize', default=640, type=int,help='detectron2 default')
    parser.add_argument('--iou', default=0.6, type=float,help='')
    parser.add_argument('--thresh', default=0.5, type=float,help='')
    parser.add_argument('--lr', default=0.00005, type=float,help='')
    parser.add_argument('--test', action = 'store_true',help='')
    parser.add_argument('--img_fmt', default='npy', type=str,help='')
    parser.add_argument('--dataset_train', default='/home/users/l/lastufka/scratch/MiraBest') #MGCLS_data/enhanced/test_data_prep_cs/train', help='')

    #parser.add_argument('--cuda', default=['0'], nargs='+', help='')
    args = parser.parse_args()
    return args

def get_dataset(args, transform = None):
    if args.test:
        train = False
    else:
        train = True
    if "MGCLS" in args.dataset_train:
        dataset = MGCLSodDataset(None, None, train=train, transform=transform, ext='.npy')
    elif "rgz" in args.dataset_train:
        dataset = RGZodDataset(None, None, train = train, transform = transform)
    elif "MNIST" in args.dataset_train:
        dataset = GalaxyMNIST(root=args.dataset_train, download = False, train=train)
    print(f"Data loaded with {len(dataset)} imgs.")
    return dataset        

def run_main(args):
    pool = False
    if "dinov2" in args.model_name:
        pool = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = pipeline(task="image-feature-extraction", model=args.model_name, device=DEVICE, pool=pool)
    dataset_train = get_dataset(args)
    all_outputs = []
    if args.img_fmt == 'npy':
        data_loader = torch.utils.data.DataLoader(
            dataset_train,
            #sampler=sampler,
            batch_size=args.ims_per_batch,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        for batch in iter(data_loader):
            if len(batch) == 2: #has labels
                pils = [to_pil_image(b) for b in batch[0]]
            else:
                pils = [to_pil_image(b) for b in batch]
            out = pipe(pils)
            for o in out:
                o = np.array(o).squeeze()
                if o.ndim == 3:
                    o = np.mean(o, axis=1)
                all_outputs.append(o)
    
    else:
        out = pipe(dataset_train)
        for o in out:
            o = np.array(o).squeeze()
            if o.ndim == 4:
                o = np.mean(o, axis=2) 
            all_outputs.append(o)
    
    all_outputs = np.array(all_outputs)
    print(all_outputs.shape)
    torch.save(torch.from_numpy(all_outputs), os.path.join(args.output_dir,"features.pth"))
    #return outputs

if __name__ == "__main__":
    args = get_args()
    run_main(args)