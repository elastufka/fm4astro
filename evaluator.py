import glob
import torch
import os
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import wandb
#from feuerzeug.utils import DINO_log_to_df, fit_pca, norm_label
from classifier_model import *
from fm_datasets import RegressionDataset
#from torcheval.metrics import MeanSquaredError, R2Score
#from astro_all_purpose.MeerKAT_utils import *
#from eval_dino import label_mask, do_sim_search, build_faiss_idx
import yaml
import argparse
import copy

def norm_label(lab):
    #return (lab - np.nanmean(lab))/np.nanstd(lab)
    p2,p98 = np.nanpercentile(lab,(2,98))
    labarr = np.array(lab)
    labarr[np.where(labarr <=p2)] = p2
    labarr[np.where(labarr >=p98)] = p98
    return (labarr-p2)/(p98-p2)

def fit_pca(data, labels=None, n_components=15):
    pca = PCA(n_components=n_components)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    components = pipe.fit_transform(data)
    if isinstance(labels,np.ndarray):
        idx = np.array(labels).reshape(len(components),1)
    else:
        idx = np.arange(len(data)).reshape(len(data),1)
    comp = np.hstack([components,idx])
    labels = {str(i): f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
    siglabels = sum([1 for i,var in enumerate(pca.explained_variance_ratio_) if var > 0.01])
    #print(np.cumsum(pca.explained_variance_ratio_))
    return comp, labels, siglabels

def default_config(output_dir, arch="vit_small", ep = 100, all = False):
    tags = {"pre-trained backbone": arch,
            "pre-training epochs": ep,
            "pre-training dataset": "MGCLS_5k",
            "sigma": 3,
            "pca": 30,
            "DQF": None,
            "augmentations": "standard",
            "datasets":[{"classify":"MiraBest"}]}
    tasks = {#"log":{"input":"log.txt"},
             #"tsne":{"input":[f"trainfeat_{arch}_{ep}.pth"],"labels":["/home/users/l/lastufka/scratch/MGCLS_data/basic/crops_256/crops_256_full_metadata2.csv"]},
            #"linear":{"input":[f"trainfeat_{arch}_{ep}.pth",f"mightee_trainfeat_{arch}_{ep}.pth"],"labels":["/home/users/l/lastufka/scratch/MGCLS_data/basic/crops_256/crops_256_full_metadata2.csv","/home/users/l/lastufka/scratch/MIGHTEE/early_science/MIGHTEE_crops_224_metadata.csv"],"quant":"iscrowd_count","epochs":100},
             "classify":{"input":[{"train":f"mb_trainfeat_{arch}_{ep}.pth","test":f"mb_testfeat_{arch}_{ep}.pth"},{"train":f"mighteeFR_trainfeat_{arch}_{ep}.pth","test":None},{"train":f"FIRST_trainfeat_{arch}_{ep}.pth","test":f"FIRST_testfeat_{arch}_{ep}.pth"}], "labels":["/home/users/l/lastufka/scratch/MiraBest/labels_confident.csv","/home/users/l/lastufka/scratch/MIGHTEE/extended_gz_catalog.csv","/home/users/l/lastufka/scratch/FIRSTGalaxyData/train_test_labels.csv"], "hps":{"lr":0.005,"seed":14,"batch_size":16, "epochs":400}},
             #"similarity":{"input":[{"train":f"mb_trainfeat_{arch}_{ep}.pth","test":f"mb_testfeat_{arch}_{ep}.pth"},{"train":f"mighteeFR_trainfeat_{arch}_{ep}.pth","test":None}], "labels":["/home/users/l/lastufka/scratch/MiraBest/labels_confident.csv","/home/users/l/lastufka/scratch/MIGHTEE/extended_gz_catalog.csv"]}
             }
    #f"mightee_trainfeat_{arch}_{ep}.pth"
    #"/home/users/l/lastufka/scratch/MIGHTEE/early_science/MIGHTEE_crops_224_metadata.csv"
    #{"train":f"mb_trainfeat_{arch}_{ep}.pth","test":f"mb_testfeat_{arch}_{ep}.pth"}
    #,"/home/users/l/lastufka/scratch/MiraBest/labels_confident.csv"
    #tags['datasets'] = [{k:tasks[k]["input"]} for k in tasks.keys() if k != 'log']
    config = {"output_dir":output_dir,"tasks": tasks, "tags":tags, "all": all}
    return config

def get_all_configs(config):
    """Given a starting config, generate configs for all feature files found in output_dir or subdirs of output_dir"""
    output_dir = config['output_dir']
    inps = [] #all task inputs, to get the naming scheme
    configs = []
    arch = config["tags"]["pre-trained backbone"] #assuming this stays the same
    for task in config['tasks'].keys():
        if task != 'log':
            inp = config['tasks'][task]['input']
        try:
            inp = inp[0]['train']
        except TypeError:
            inp = inp[0]
        inps.append(inp)
    headers = [inp[:inp.find(arch)] for inp in inps]
    
    featfiles = sorted(glob.glob(os.path.join(output_dir,"trainfeat*.pth")))
    
    if len(featfiles) == 0:
        #only subdirs in the path
        subdirs = sorted(glob.glob(os.path.join(output_dir,"*")))
        featfiles = [os.path.join(s,inps[0]) for s in subdirs]
    
    for f in featfiles:
        new_config = copy.deepcopy(config)
        #update tags
        epochs = int(f[f.find(arch)+len(arch)+1:-4])
        new_config['tags']['pre-training epochs'] = epochs
        str_epochs = str(epochs).zfill(3)
        #augmentations?
        
        for head, task in zip(headers, new_config['tasks'].keys()):
            if task == "linear":
                new_config['tasks'][task]['input'] = [f"{head}{arch}_{str_epochs}.pth"]
            elif task in ["classify", "similarity"]:
                try:
                    testfile = new_config['tasks'][task]['input'][0]['test'] 
                except KeyError:
                    testfile = new_config['tasks'][task]['input']['test']
                if testfile is not None:
                    testfile = f"{testfile[:testfile.find(arch)]}{arch}_{str_epochs}.pth"
                new_config['tasks'][task]['input'] = [{'train': f"{head}{arch}_{str_epochs}.pth", 'test': testfile}]
        configs.append(new_config)
        #print(new_config)
        del new_config
    return configs

class Evaluator():
    def __init__(self, config, project, entity='elastufka', log_best = False):
        self.config = config
        self.project = project
        self.entity = entity
        self.log_best = log_best 
        self.verbose = not log_best
        self.config['tags']['verbose'] = self.verbose
        self.output_dir = config["output_dir"]
        self._set_task_flags()
        
    def _set_task_flags(self):
        #self.log = True if "log" in self.config["tasks"].keys() else False
        #self.tnse = True if "tsne" in self.config["tasks"].keys() else False
        #self.linear = True if "linear" in self.config["tasks"].keys() else False
        self.classify = True if "classify" in self.config["tasks"].keys() else False 
        #self.similarity = True if "similarity" in self.config["tasks"].keys() else False
            
    def _init_wandb(self):
        config = self.config['tags']
        for task in self.config["tasks"]:
            if 'hps' in self.config["tasks"][task].keys():
                for k,v in self.config["tasks"][task]['hps'].items():
                    config[k] = v
            config[f"{task}_input"] = self.config["tasks"][task]["input"]
            config[f"{task}_labels"] = self.config["tasks"][task]["labels"]
        wandb.init(project=self.project, config=config, entity = self.entity) 
        
    def evaluate(self):
        self._init_wandb()
        # if self.log:
        #     self.log_training()
        # if self.tnse:
        #     self.do_tsne()
        # if self.linear:
        #     self.linear_eval()
        if self.classify:
            self.class_eval()
        # if self.similarity:
        #     self.sim_eval()
        wandb.finish()
            
    # def log_training(self):
    #     logfile = os.path.join(self.output_dir, self.config['tasks']['log']['input'])
    #     if "MSN" in self.config['tags']['pre-trained backbone']:
    #         ldf = MSN_log_to_df(log)
    #     else:
    #         ldf = DINO_log_to_df(logfile)
    #     ldf = ldf.drop_duplicates(subset='epoch', keep='last')
    #     ldf = ldf.where(ldf.epoch <= self.config['tags']['pre-training epochs']).dropna(how='all')

    #     for _, row in ldf.iterrows():
    #         wandb.log({"pretrain_epoch":row.epoch, "Avg. pretrain_loss":row.train_loss})

    # def do_tsne(self):
    #     """if multiple inputs, do them together..."""
    #     all_dfs = []
    #     for i,l in zip(self.config['tasks']['tsne']['input'], self.config['tasks']['tsne']['labels']):
    #         n_components = 30 if self.config['tags']['pca'] == 0 else self.config['tags']['pca']
    #         feats = torch.load(os.path.join(self.output_dir, i)).cpu().detach().numpy()
    #         comp, _, _ = fit_pca(feats, n_components=n_components)
    #         labels = pd.read_csv(l)
    #         cdict = {}
    #         #cdict['source_name'] = labels['source_name']
    #         cdict['input'] = i
    #         for k in labels.keys():
    #             if k in ["iscrowd_count","area_mean","area_sum","z","rms (μJy beam−1)"]:
    #                 if labels[k].fillna(0).std() != 0.0:
    #                     cdict[k] = ml_vis.norm_label(labels[k].fillna(0))
    #             else:
    #                 cdict[k] = labels[k].fillna(0)

    #         for i in range(comp.shape[-1]):
    #             cdict[f"PCA_comp{i}"] = comp[:,i]
    #         cdf = pd.DataFrame(cdict)
    #         all_dfs.append(cdf)
    #     edf = pd.concat(all_dfs)
    #     wandb.log({"embedding":edf})

    # def linear_eval(self):
    #     quant = self.config['tasks']['linear']['quant']
    #     num_epochs = self.config['tasks']['linear']["epochs"]
    #     dqf = self.config['tags']['DQF']
    #     unseen = self.config['tags']['unseen']
    #     sigma = self.config['tags']['sigma']
    #     split = self.config['tasks']['linear']['split']
                
    #     for ffile, lfile in zip(self.config['tasks']['linear']['input'], self.config['tasks']['linear']['labels']):
    #         if ffile.endswith(".pth"):
    #             feats = torch.load(os.path.join(self.output_dir, ffile)).cpu()
    #         else: 
    #             feats = np.load(os.path.join(self.output_dir, ffile))
    #         if self.config['tags']['pca'] !=0:
    #             feats, _, _ = fit_pca(feats, n_components = self.config['tags']['pca'])

    #         labels = self._get_crop_labels(lfile)
    #         y = labels[quant].values
    #         nanidx = np.where(np.isnan(y))[0]
    #         X_scaled = StandardScaler().fit_transform(feats)
    #         X_scaled= np.delete(X_scaled,nanidx,axis=0)
    #         y = np.delete(y,nanidx,axis=0)
    #         meta = labels.drop(nanidx).reset_index(drop=True) 
                
    #         if dqf is not None:
    #             X_scaled, y, meta = self._select_dqf(X_scaled, y, meta, dqf = dqf) #don't know how this combines with sigma
        
    #         if unseen:
    #             meta = meta.where(meta.in_subset == 0).dropna(how = 'all')
    #             dqi = meta.index
    #             X_scaled = X_scaled[dqi]
    #             y = y[dqi]
                
    #         if sigma != 0:
    #             lmask = label_mask(meta, sigma = sigma)
    #             #print(np.unique(lmask), np.sum(lmask))
    #             #print(np.sum(~lmask))
    #             yrest = y[~lmask]
    #             X_rest = X_scaled[~lmask,:]
    #             y = y[lmask]
    #             X_scaled = X_scaled[lmask,:]
    #             #print(y.shape,X_scaled.shape)
    #             #print(yrest.shape,X_rest.shape)

    #         dataset = RegressionDataset(X_scaled, y)
            
    #         train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[0.7,0.2,0.1], generator=torch.Generator().manual_seed(self.config['tasks']['linear']['seed'])) #what if no classify?
            
    #         train_losses, val_losses, model  = run_linregress(train_dataset, val_dataset, num_epochs = num_epochs, verbose = self.verbose)
    #         #if args.save_model is not None and nn == "MGCLS":
    #         #    torch.save(model.state_dict(),os.path.join(args.save_model, "linear_model.pth"))
    #         if not split:
    #             test_dataset = dataset
    #         mets, y_pred, y_true = eval_linregress(test_dataset, None, model) #need to get all...
    #         print(f"Linear regression with {len(test_dataset)} frozen features from {ffile} to {quant}, DQF={dqf}")
    #         print(f"MSE: {mets[0]:.3f}") #\n R2: {mets[1]:.3f}")
    #         try:
    #             dataset = self.config['tags']['datasets'][0]['linear']
    #         except KeyError:
    #             dataset = self.config['tags']['datasets']['linear']
    #         wandb.log({"pretrain_epoch":self.config['tags']['pre-training epochs'],f"{dataset} MSE":mets[0]})

    #         if self.verbose:
    #             title = "MGCLS source count prediction" if "MGCLS" in lfile else "MIGHTEE source count prediction"
    #             table = wandb.Table(data = [[t,p] for (t,p) in zip(y_true, y_pred)], columns = ["true","predicted"])
    #             wandb.log({"linear_results_test" : wandb.plot.scatter(table, "true","predicted", title = title)})
                
    #             metstr, y_predtr, y_truetr = eval_linregress(train_dataset, None, model)
    #             print(f"Linear regression with {len(train_dataset)} train set from {ffile} to {quant}, DQF={dqf}")
    #             print(f"MSE: {metstr[0]:.3f}") #\n R2: {mets[1]:.3f}")
    #             table = wandb.Table(data = [[t,p] for (t,p) in zip(y_truetr, y_predtr)], columns = ["true","predicted"])
    #             wandb.log({"linear_results_train" : wandb.plot.scatter(table, "true","predicted", title = title)})

    #             metsv, y_predv, y_truev = eval_linregress(val_dataset, None, model)
    #             print(f"Linear regression with {len(val_dataset)} val set from {ffile} to {quant}, DQF={dqf}")
    #             print(f"MSE: {metsv[0]:.3f}") #\n R2: {mets[1]:.3f}")
    #             table = wandb.Table(data = [[t,p] for (t,p) in zip(y_truev, y_predv)], columns = ["true","predicted"])
    #             wandb.log({"linear_results_val" : wandb.plot.scatter(table, "true","predicted", title = title)})
                
    #             if sigma != 0:
    #                 rest_dataset = RegressionDataset(X_rest, yrest)
    #                 metsr, y_predr, y_truer = eval_linregress(rest_dataset, None, model)
    #                 print(f"Linear regression with {len(yrest)} outliers from {ffile} to {quant}, DQF={dqf}")
    #                 print(f"MSE: {metsr[0]:.3f}") #\n R2: {mets[1]:.3f}")
    #                 table = wandb.Table(data = [[t,p] for (t,p) in zip(y_truer, y_predr)], columns = ["true","predicted"])
    #                 wandb.log({"linear_results_rest" : wandb.plot.scatter(table, "true","predicted", title = title)})
    #         del model

                
    def class_eval(self):
        hps = self.config['tasks']['classify']['hps']
        dense_layers = hps['dense_layers']
        lkey = hps['lkey']
        bottleneck = hps['bottleneck']
        single_object = hps['single_object']
        dropout = hps['dropout']
        confmat = hps['confmat']
        class_names = hps['class_names']
        class_weights = hps['class_weights']
        num_classes = self._get_num_classes(lkey=lkey)
        last_layer = self.config['tags']['last_layer']
        nlabels = hps['n_train_labels']
        print("num classes", num_classes,"lkey", lkey)
        
        for ffile, lfile in zip(self.config['tasks']['classify']['input'], self.config['tasks']['classify']['labels']):
            trainfile = ffile["train"]
            testfile = ffile["test"]
            train_dataset, test_dataset = self._get_FR_feats(trainfile, testfile, lfile, hps['seed'], nlabels, lkey =lkey,single_object = single_object)

            if nlabels is not None:
                full_train_dataset, _ = self._get_FR_feats(trainfile, testfile, lfile, hps['seed'], None, lkey =lkey,single_object = single_object)
                full_train_loader = DataLoader(dataset = full_train_dataset, batch_size = hps['batch_size'])

            train_loader = DataLoader(dataset = train_dataset, batch_size = hps['batch_size'], shuffle = True)
            test_loader = DataLoader(dataset = test_dataset, batch_size = hps['batch_size'], shuffle = True)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if class_weights is not None:
                class_weights = torch.Tensor(class_weights).to(device)
            try:
                in_feats = train_dataset.X.shape[1] 
            except AttributeError:
                train_dataset.dataset.X.shape[1] 
            if num_classes == 2:
                criterion = nn.BCELoss(weight=class_weights)
                eval_fn = eval_binary
                nclass = 1
                #class_names = ["FRII","FRI"]
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                eval_fn = eval_multiclass
                nclass = num_classes
                #class_names = ["FRI","FRII", "C","B"]
            #sclass = LogisticRegression(in_feats, nclass, n_layers = dense_layers, bottleneck = bottleneck, dropout = dropout).to(device)
            sclass = LinearClassifierLayer(in_feats, num_labels=nclass).to(device)

            optimizer = torch.optim.SGD(sclass.parameters(), lr = hps['lr']) #change from ADAM to avoid overfitting

            acc, frmodel = train_and_eval_model(sclass, train_loader, test_loader, criterion, optimizer, eval_fn, n_eval = 1, n_epochs=hps['epochs'], verbose=self.verbose, confmat=False, keep_best_acc=self.log_best, class_names = class_names)
            print(f"Final model F1 score: {acc}") #this should be the only thing logged if '--all'
            
            try:
                #dataset = self.config['tags']['datasets'][0]['classify']
                dataset = self.config['tags']['datasets']['classify']
            except TypeError:
                for d in self.config['tags']['datasets']:
                    if "classify" in d.keys():
                        dataset = d['classify']
                        break
            if not self.verbose:
                #print(acc)
                #print([type(a) for a in acc])
                if isinstance(acc[1], torch.Tensor) and acc[1].dim() > 0:
                    ldict = {"pretrain_epoch":self.config['tags']['pre-training epochs'],f"{dataset} final loss": acc[0]}
                    for i,cn in enumerate(class_names):
                        ldict[f"accuracy_{cn}"] = acc[1][i].numpy()
                        ldict[f"F1 score_{cn}"] = acc[2][i].numpy()
                        ldict[f"precision_{cn}"] = acc[3][i].numpy()
                        ldict[f"recall_{cn}"] = acc[4][i].numpy()
                    ldict[f"accuracy"] = acc[1].mean(axis=1).numpy() #do i need to average
                    ldict[f"F1 score"] = acc[2].mean(axis=1).numpy()
                    ldict[f"precision"] = acc[3].mean(axis=1).numpy()
                    ldict[f"recall"] = acc[4].mean(axis=1).numpy()
                    print(ldict)
                    wandb.log(ldict)
                else:
                    wandb.log({"pretrain_epoch":self.config['tags']['pre-training epochs'],f"{dataset} final loss": acc[0], f"{dataset} accuracy": acc[1], f'{dataset} F1 score': acc[2],f'{dataset} precision':acc[3],f'{dataset} recall':acc[4]})
            
            if confmat and len(class_names) > 2:
                _, predictions, ground_truth = eval_fn(test_loader, frmodel, device=device, n_classes=len(class_names), return_vals=True) 
                wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=class_names)})

            if last_layer:
                if nlabels is not None:
                    ltrain, trainl = get_last_layer(frmodel, full_train_loader)
                else:
                    ltrain, trainl = get_last_layer(frmodel, train_loader)
                ltest, testl = get_last_layer(frmodel, test_loader)
                ll = torch.concat([ltrain, ltest])
                fn = os.path.join(self.config["output_dir"], f"{self.config['tags']['pre-trained backbone']}_class_ll_{nlabels}.pth")
                tfn = os.path.join(self.config["output_dir"], f"{self.config['tags']['pre-trained backbone']}_class_ll_labels_{nlabels}.pth")
                torch.save(ll, fn)
                #save labels too
                labs = torch.concat([trainl, testl]).numpy()
                ldf = pd.DataFrame({'label':labs})
                ldf.to_csv(tfn)
                #save it or go directly to TSNE
                self.config['tasks']['tsne'] = {'input':[fn],'labels': [tfn]}
                self.do_tsne()
                
            del sclass
            del frmodel
            

    # def sim_eval(self):
    #     hps = self.config['tasks']['classify']['hps']
    #     for ffile, lfile in zip(self.config['tasks']['similarity']['input'], self.config['tasks']['similarity']['labels']):
    #         trainfile = ffile["train"]
    #         testfile = ffile["test"]
    #         train_feats, train_labels, test_feats, test_labels = self._get_FR_feats(trainfile, testfile, lfile, hps['seed'], return_dataset = False)
        
    #         index = build_faiss_idx(train_feats)
    #         #print(f"feature shapes {len(train_feats)},{len(test_feats)}")
    #         #try:
    #         #    dataset = self.config['tags']['datasets'][0]['similarity']
    #         #except KeyError:
    #         #    dataset = self.config['tags']['datasets']['similarity']
                
    #         try:
    #             #dataset = self.config['tags']['datasets'][0]['classify']
    #             dataset = self.config['tags']['datasets']['similarity']
    #         except TypeError:
    #             for d in self.config['tags']['datasets']:
    #                 if "similarity" in d.keys():
    #                     dataset = d['similarity']
    #                     break
    #         for t in [1,3,5]:
    #             mbss = do_sim_search(test_feats, test_labels, index, train_labels, 5, top = t)
    #             print(f"{mbss*100:.2f}% of labels match within top {t} results\n")
    #             wandb.log({"pretrain_epoch": self.config['tags']['pre-training epochs'], f"{dataset} Precision@{t}":mbss})


    def _get_num_classes(self, return_values = False, lkey = 'simple_labels',single_object=False):
        labels = pd.read_csv(os.path.join(self.output_dir,self.config['tasks']['classify']['labels'][0]))
        if single_object:
            l0 = lkey
            lkey = [l0,"n_annotations"]
        if "has_nan" in labels.keys(): # or np.isnan(sum(labels[lkey].unique())):
            good_labels = labels.where(labels.has_nan == 0).dropna(how='all')
            train_labels = good_labels.majority_classification.values
            y = train_labels
        if "split" in labels.keys():
            train_labels = labels.where(labels.split == 'train').dropna(how='all')[lkey]#.values 
            test_labels = labels.where(labels.split == 'test').dropna(how='all')[lkey]#.values 
            if single_object: #only get where n_annotations == 1
                train_labels = train_labels.where(train_labels.n_annotations == 1)[l0].dropna(how='all')
                test_labels = test_labels.where(test_labels.n_annotations == 1)[l0].dropna(how='all')
            else:
                #print("train", len(train_labels), "test", len(test_labels))
                train_labels = train_labels.fillna(-1).values
                test_labels = test_labels.fillna(-1).values
                #print(np.unique(train_labels))
            try:
                #train_labels = #labels.where(labels.split == 'train').dropna(how='all')[lkey].values.astype(np.int64) #
                y = train_labels.astype(np.int64)
            except TypeError: #test this
                y = train_labels
        else:
            y = labels[lkey].values
        #print(np.unique(y))
        if isinstance(y[0], str):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y) #0 = FRII, 1 = FR1

        print(single_object,"train", len(train_labels), "test", len(test_labels))
        if return_values and "split" in labels.keys():
            return y, test_labels
        elif return_values:
            return y, None

        if np.unique(y)[0] < 0:
            return len(np.unique(y)) - 1    
        else:
            return len(np.unique(y))
        
    def _get_crop_labels(self, lfile):
        cropdf = pd.read_csv(lfile)
        try:
            labels = cropdf[["file_prefix_basic","OBSRA_basic","OBSDEC_basic","iscrowd_count","iscrowd_sum","area_mean","area_sum","DATAMEAN_basic","z","rms (μJy beam−1)","DQF"]]
        except KeyError:
            labels = cropdf[["file_prefix","OBSRA","OBSDEC","iscrowd_count","iscrowd_sum","area_mean","area_sum","DATAMEAN","z","rms (μJy beam−1)","DQF"]]
        try:
            labels["in_subset"] = cropdf["in_subset"].values
        except KeyError:
            pass
        return labels
    
    def _select_dqf(self, comp, labels, meta, dqf = None):
        if dqf is not None:
            meta = meta.where(meta.DQF == dqf).dropna(how = 'all')
            dqi = meta.index
            return comp[dqi], labels[dqi], meta
        else:
            return comp, labels, meta
        
        
    def _get_FR_feats(self, trainfile, testfile, lfile, seed, nlabels, return_dataset=True, lkey = "simple_labels", single_object=False):
        if trainfile.endswith(".pth"):
            trainfeats = torch.load(os.path.join(self.output_dir,trainfile)).cpu() 
        else: 
            trainfeats = np.load(os.path.join(self.output_dir,trainfile))
        if self.config['tags']['pca'] is not None:
            trainfeats, _, _ = fit_pca(trainfeats, n_components = self.config['tags']['pca'])
        #labels
        train_labels, test_labels = self._get_num_classes(return_values = True, lkey = lkey, single_object = single_object)
        #full_labels = pd.read_csv(os.path.join(self.output_dir,lfile))

        if single_object: #train_labels and test_labels is Series
            trainidx = train_labels.index
            testidx = test_labels.index - len(trainfeats)

            trainfeats = trainfeats[trainidx]
            train_labels = train_labels.values
            test_labels = test_labels.values
            
        y = train_labels # pd.read_csv(os.path.join(self.output_dir,lfile))
        #if return_dataset: #remove not
        #scale and encode
        X_scaled = StandardScaler().fit_transform(trainfeats)
        #print(X_scaled.shape)
        #else:
        #    X_scaled = np.array(trainfeats)
        #construct datasets
        dataset = RegressionDataset(X_scaled, y)
        if testfile is None: #not isinstance(testfile, str):
            train_dataset, test_dataset = random_split(dataset, lengths=[0.8,0.2], generator=torch.Generator().manual_seed(seed))
            #check dataset balance
            train_feats = train_dataset.dataset.X.numpy()[train_dataset.indices]
            test_feats = test_dataset.dataset.X.numpy()[test_dataset.indices]
            train_labels = train_dataset.dataset.y.numpy()[train_dataset.indices]
            test_labels = test_dataset.dataset.y.numpy()[test_dataset.indices]

        else:
            #print("has test feats")
            train_feats = X_scaled
            if testfile.endswith('.pth'):
                test_feats = torch.load(os.path.join(self.output_dir,testfile)).cpu()
            else: 
                test_feats = np.load(os.path.join(self.output_dir,testfile))
            if single_object:
                test_feats = test_feats[testidx]
            if self.config['tags']['pca'] is not None:
                test_feats, _, _ = fit_pca(test_feats, n_components = self.config['tags']['pca'])
            #if return_dataset: #remove not
            #scale and encode
            test_feats = StandardScaler().fit_transform(test_feats)
            #else:
            #    test_feats = np.array(test_feats)
        print(train_feats.shape, test_feats.shape)
        assert len(train_feats) == len(train_labels)

        if np.unique(train_labels)[0] == -1: #delete feats where label is nan
            nanarr = np.argwhere(train_labels == -1).squeeze()
            train_feats = np.delete(train_feats, nanarr, axis=0)
            train_labels = np.delete(train_labels, nanarr, axis=0)
            nanarr = np.argwhere(test_labels == -1).squeeze()
            test_feats = np.delete(test_feats, nanarr, axis=0)
            test_labels = np.delete(test_labels, nanarr, axis=0)

        if nlabels: #None if all
            i = 0
            lunique = 0
            
            while lunique != len(np.unique(test_labels)):
                np.random.seed(seed+i)
                random_idx = np.random.choice(len(train_feats), size = nlabels, replace = False)
                train_feats = train_feats[random_idx]
                train_labels = train_labels[random_idx]
                i += 1
                lunique = len(np.unique(train_labels))
            print(f"Not all classes were represented with random seed {seed}, therefore new seed {seed+i} was used.")

        train_dataset = RegressionDataset(train_feats, train_labels)
        test_dataset = RegressionDataset(test_feats, test_labels)

        if return_dataset:
            return train_dataset, test_dataset
        else:
            return train_feats, train_labels, test_feats, test_labels
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation tasks')
    parser.add_argument('--config', default=None, type=str, help='config yaml file') #"/home/users/l/lastufka/mightee_dino/eval.yaml"
    parser.add_argument('--project', default="test_eval", type=str, help='name of W&B project')
    parser.add_argument('--log_best', action="store_true", help='name of W&B project')
    args = parser.parse_args()
    
    if not args.config:
        #get from environ
        config = default_config(os.environ["output_dir"],arch=os.environ["arch"], ep=os.environ["epochs"])
        #config["output_dir"] = os.environ["output_dir"]
        config["tags"]["pre-training dataset"] = os.environ["pretrain_data"]
        config["tags"]["augmentations"] = os.environ["augs"]
        del config['tasks']['log']
        del config['tasks']['tsne']
        
        #do it twice... 
        config_mgcls = copy.deepcopy(config)
        config_mgcls['tags']['datasets'] = {"linear":"MGCLS","classify":"MiraBest","similarity":"MiraBest"}
        config_mightee = copy.deepcopy(config)
        config_mightee['tags']['datasets'] = {"linear":"MIGHTEE","classify":"MIGHTEE","similarity":"MIGHTEE"}
 
        for t in config['tasks'].keys():
            if t not in ["log","tsne"]:
                #print(t, len(config["tasks"][t]["input"]))
                config_mgcls["tasks"][t]["input"] = [config["tasks"][t]["input"][0]]
                config_mgcls["tasks"][t]["labels"] = [config["tasks"][t]["labels"][0]]
                config_mightee["tasks"][t]["input"] = [config["tasks"][t]["input"][1]]
                config_mightee["tasks"][t]["labels"] = [config["tasks"][t]["labels"][1]]
        for config in [config_mgcls, config_mightee]:
            ev = Evaluator(config, args.project, log_best = bool(args.log_best))
            print(config)
            ev.evaluate()
        
    else:    
        with open(args.config, "r") as y:
            config = yaml.load(y, Loader = yaml.FullLoader)
        if config['all']:
            configs = get_all_configs(config)
            for config in configs:
                ev = Evaluator(config, args.project, log_best = True)
                print(config)
                ev.evaluate()
        else:
            ev = Evaluator(config, args.project, log_best = args.log_best)
            print(ev.log_best, ev.verbose)
            reps = config['reps']
            if reps is not None:
                for i in range(reps):
                    ev.evaluate()
            else: 
                ev.evaluate()

