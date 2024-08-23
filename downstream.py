import argparse
#import sys
#sys.path.append('/home/users/l/lastufka')
from feuerzeug.models import *
from feuerzeug.evaluator import Evaluator
import yaml
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/glados/unix-Documents/AstroSignals/fm_compare/eval.yaml", type=str,help='')
    parser.add_argument('--output_dir', default="/home/glados/unix-Documents/AstroSignals/fm_compare", type=str,help='')
    parser.add_argument('--project', default="FM_compare", type=str,help='')
    parser.add_argument('--ims_per_batch', default=8, type=int,help='')
    parser.add_argument('--imsize', default=640, type=int,help='detectron2 default')
    parser.add_argument('--iou', default=0.6, type=float,help='')
    parser.add_argument('--thresh', default=0.5, type=float,help='')
    parser.add_argument('--lr', default=0.00005, type=float,help='')
    parser.add_argument('--log_best', action = 'store_true',help='')
    parser.add_argument('--img_fmt', default='npy', type=str,help='')
    parser.add_argument('--dataset_train', default='/home/users/l/lastufka/scratch/MiraBest') #MGCLS_data/enhanced/test_data_prep_cs/train', help='')

    #parser.add_argument('--cuda', default=['0'], nargs='+', help='')
    args = parser.parse_args()
    return args

def get_all_configs(config):
    """Given a starting config, generate configs for all feature files found in output_dir or subdirs of output_dir"""
    output_dir = config['output_dir']
    inps = [] #all task inputs, to get the naming scheme
    configs = []
    #arch = config["tags"]["pre-trained backbone"] #assuming this stays the same
    # for task in config['tasks'].keys():
    #     if task != 'log':
    #         inp = config['tasks'][task]['input']
    #     try:
    #         inp = inp[0]['train']
    #     except TypeError:
    #         inp = inp[0]
    #     inps.append(inp)
    # headers = [inp[:inp.find(arch)] for inp in inps]
    
    trainfiles = sorted(glob.glob(os.path.join(output_dir,"*trainfeat*mean.pth")))
    testfiles = sorted(glob.glob(os.path.join(output_dir,"*testfeat*mean.pth")))
    trainfiles = [t[t.rfind("/")+1:] for t in trainfiles]
    testfiles = [t[t.rfind("/")+1:] for t in testfiles]
    #if len(trainfiles) == 0:
    #    #only subdirs in the path
    #    subdirs = sorted(glob.glob(os.path.join(output_dir,"*")))
    #    featfiles = [os.path.join(s,inps[0]) for s in subdirs]
    
    for tf,ft in zip(trainfiles, testfiles):
        new_config = copy.deepcopy(config)
        #update tags
        #epochs = int(f[f.find(arch)+len(arch)+1:-4])
        arch = tf[tf.find("feat_")+5:tf.find("mean")-1]
        print(arch)
        new_config['tags']['pre-trained backbone'] = arch
        #str_epochs = str(epochs).zfill(3)
        #augmentations?
        
        for task in new_config['tasks'].keys():
            if task == "linear":
                new_config['tasks'][task]['input'] = [f"{head}{arch}_{str_epochs}.pth"]
            elif task in ["classify", "similarity"]:
                #try:
                #    testfile = new_config['tasks'][task]['input'][0]['test'] 
                #except KeyError:
                #    testfile = new_config['tasks'][task]['input']['test']
                #if testfile is not None:
                 #   testfile = f"{testfile[:testfile.find(arch)]}{arch}_{str_epochs}.pth"
                new_config['tasks'][task]['input'] = [{'train': tf, 'test': ft}]
        configs.append(new_config)
        #print(new_config)
        del new_config
    return configs

def train_projector(args):
    with open(args.config, "r") as y:
        config = yaml.load(y, Loader = yaml.FullLoader)
    # if config['all']:
    #     configs = get_all_configs(config)
    #     for config in configs:
    #         ev = Evaluator(config, args.project, log_best = True)
    #         print(config)
    #         ev.evaluate()
    # else:
    reps = config['reps']
    if config['all']:
        configs = get_all_configs(config)
        print(configs)
        for c in configs:
            ev = Evaluator(c, args.project, log_best = args.log_best)
            if reps is not None:
                for i in range(reps):
                    ev.evaluate()
            else: 
                ev.evaluate()

    ev = Evaluator(config, args.project, log_best = args.log_best)
    if reps is not None:
        for i in range(reps):
            ev.evaluate()
    else: 
        ev.evaluate()

if __name__ == "__main__":
    args = get_args()
    train_projector(args)