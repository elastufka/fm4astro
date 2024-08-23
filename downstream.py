import argparse
from evaluator import Evaluator
import yaml
import copy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="eval.yaml", type=str,help='')
    parser.add_argument('--output_dir', default=".", type=str,help='')
    parser.add_argument('--project', default="FM_compare", type=str,help='')
    parser.add_argument('--ims_per_batch', default=8, type=int,help='')
    parser.add_argument('--imsize', default=640, type=int,help='detectron2 default')
    parser.add_argument('--iou', default=0.6, type=float,help='')
    parser.add_argument('--thresh', default=0.5, type=float,help='')
    parser.add_argument('--lr', default=0.00005, type=float,help='')
    parser.add_argument('--log_best', action = 'store_true',help='')
    parser.add_argument('--img_fmt', default='npy', type=str,help='')
    parser.add_argument('--dataset_train', default='.') #MGCLS_data/enhanced/test_data_prep_cs/train', help='')

    #parser.add_argument('--cuda', default=['0'], nargs='+', help='')
    args = parser.parse_args()
    return args

def get_all_configs(config):
    """Given a starting config, generate configs for all feature files found in output_dir or subdirs of output_dir"""
    output_dir = config['output_dir']
    inps = [] #all task inputs, to get the naming scheme
    configs = []

    if config['tags']['datasets'][0]['classify'] == 'RGZ':
        plabels = [469,1408,2346] 
    #elif config['tags']['datasets'][0]['classify'] == 'GZ10':
    #    plabels = [1596,4788,7981]
    else:
        plabels = [800,2400,4000]

    for nl in plabels: 
        new_config = copy.deepcopy(config)
        
        for task in new_config['tasks'].keys():
            if task in ["classify", "similarity"]:
                new_config['tasks'][task]['hps']['n_train_labels'] = nl #['input'] = [{'train': tf, 'test': ft}]
        configs.append(new_config)
        #print(new_config)
        del new_config
    return configs

def train_projector(args):
    with open(args.config, "r") as y:
        config = yaml.load(y, Loader = yaml.FullLoader)
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