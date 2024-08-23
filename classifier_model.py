import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import models as torchvision_models
#from pytorch_mask_rcnn.model.mask_rcnn import MaskRCNN
import pickle
from fm_datasets import RegressionDataset
from torch.utils.data import DataLoader, Dataset, random_split
#from torcheval.metrics import MeanSquaredError, R2Score, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
#from torchmetrics.regression import MeanSquaredError
import torchmetrics as tm
import wandb
import copy
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


class LinearClassifierLayer(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        #self.linear.weight.data.normal_(mean=0.0, std=0.01)
        #self.linear.bias.data.zero_()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        if self.num_labels !=2: #cross-entropy loss takes logits
            return self.linear(x) 
        else:
            return self.sigmoid(self.linear(x))
    
class LogisticRegression(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, input_dim, output_dim, n_layers = 1, logits = False, dropout = None, bottleneck = None, last_layer = False):
        super().__init__()
        if bottleneck:
            intermed = bottleneck
        else:
            intermed = input_dim
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.n_layers = n_layers
        self.logits = logits
        self.first = nn.Linear(input_dim, intermed)
        self.linear = nn.Linear(intermed, intermed)
        self.active = nn.ReLU()
        self.last = nn.Linear(intermed, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = None
        self.last_layer = last_layer
        if dropout and dropout != 0.0:
            self.dropout = nn.Dropout(dropout) #probability

    def forward(self, x):
        # flatten
        if self.dropout:
            x = self.dropout(x)
        x = self.batchnorm(x)
        #print(x.shape)
        x = self.active(self.first(x))
        #print(x.shape)
        if self.n_layers > 1:
            for i in range(1, self.n_layers):
                x = self.active(self.linear(x))
                #print(x.shape)
        if self.last_layer:
            return x    
        x = self.last(x)
        #print(x.shape)
        if self.logits:
            return x
        else:
            return self.sigmoid(x) #need sigmoid... not softmax cuz binary
    
def get_last_layer(model, train_loader, device='cuda'):
    model.last_layer = True
    model.eval()
    out = []
    labs = []
    for x, y in train_loader:
        x = x.to(device)
        #y = y.to(device)
        #clear gradient 
        #optimizer.zero_grad()
        #make a prediction 
        z = model(x) 
        out.append(z)
        labs.append(y.cpu())
    outv = torch.concat(out)
    outl = torch.concat(labs).squeeze()
    return outv, outl

def train_and_eval_model(model, train_loader, test_loader, criterion, optimizer, eval_fn, n_epochs = 300, n_eval = 10, device='cuda', verbose=True, keep_best_acc = True, class_names = ['FRII','FRI'], confmat=True):
    """train & validate model (using accuracy)"""
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    loss_list, losses = [],[]
    af1,aac,apre,arec = [],[],[],[]

    model.train()
    #print(next(model.parameters()).device)
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x = x.cuda() #.to(device)
            #print(x.shape)
            y = y.cuda() #.to(device)
            #clear gradient 
            optimizer.zero_grad()
            #make a prediction 
            z = model(x) #originaly just model(x)
            if criterion._get_name() == "CrossEntropyLoss":
                loss = criterion(z,y.long().squeeze())
            else:
                loss = criterion(z,y) #need long for crossentropy .long().squeeze()
            # calculate gradients of parameters 
            loss.backward()
            # update parameters 
            optimizer.step()
            
            loss_list.append(loss.data)
        losses.append(loss.item())

        # evaluate accuracy at end of every n_eval epochs
        if epoch % n_eval == 0:
            model.eval()
            acc, f1, precision, recall = eval_fn(test_loader, model, device=device, n_classes=len(class_names)) #formerly eval_binary
            if verbose:
                if type(acc) == torch.Tensor and (acc.ndim > 1 or len(acc) == len(class_names)):
                    #do one for each...
                    ldict = {"epoch":epoch,"loss": loss.item()}
                    for i,cn in enumerate(class_names):
                        ldict[f"accuracy_{cn}"] = acc[i]
                        ldict[f"F1 score_{cn}"] = f1[i]
                        ldict[f"precision_{cn}"] = precision[i]
                        ldict[f"recall_{cn}"] = recall[i]
                    ldict[f"accuracy"] = acc.mean().numpy() #do i need to average
                    ldict[f"F1 score"] = f1.mean().numpy()
                    ldict[f"precision"] = precision.mean().numpy()
                    ldict[f"recall"] = recall.mean().numpy()
                else:
                    ldict = {"epoch":epoch,"loss": loss.item(), "accuracy": acc, 'F1 score': f1,'precision':precision,'recall':recall}
                #print(ldict)
                wandb.log(ldict)
            else: #keep them anyway and average last 10 epochs...
                af1.append(f1)
                aac.append(acc)
                apre.append(precision)
                arec.append(recall)

            # if keep_best_acc and f1 > best_acc:
            #     #print(f1)
            #     best_acc = f1
            #     best_weights = copy.deepcopy(model.state_dict())
            
        elif verbose:
            wandb.log({"loss": loss.item()})
    # restore model and return best accuracy
    if keep_best_acc == True:
        #model.load_state_dict(best_weights)
        #else:
        best_acc = losses[-1], acc, f1, precision, recall
        #if avg_10:
        #    best_acc = np.mean(losses[-10:]), np.mean(aac[-10:]), np.mean(af1[-10:]), np.mean(apre[-10:]), np.mean(arec[-10:])
            
    if verbose:
        metrics, y_pred, y_true = eval_fn(test_loader, model, device=device, return_vals=True, n_classes=len(class_names))
        #print(f"sklearn f1 score: {metrics}")
        #print(f"y_true: {y_true}")
        #cmv = confusion_matrix(y_true,y_pred.round()).ravel()
        #print(f"TN: {cmv[0]}\nFP: {cmv[1]}\nFN: {cmv[2]}\nTP: {cmv[3]}")
        #print(cmv)
        #wconfmat = wandb.plot.confusion_matrix(probs=None,y_true=y_true, preds=y_pred.round(), class_names=class_names)
    #if confmat:
    #    wandb.log({"conf_mat": wconfmat})
    return best_acc, model

def eval_binary(test_loader, model, device='cuda', metrics = [None], return_vals = False, n_classes=2): #BinaryAccuracy(), BinaryF1Score()
    model.eval()
    #y_pred, y_true = [], []

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_true = torch.cat((y_true, targets), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)
            #y_pred.extend(outputs.cpu().squeeze())
            #y_true.extend(targets.cpu().squeeze())

    mevals = []
    # for m in metrics:
    #     if len(np.unique(all_outputs.cpu().squeeze())) == 1: #only one class predicted
    #         #metric is None...
    #         mevals.append(None)
    #     m.update(all_outputs.cpu().squeeze(), y_true.cpu().squeeze())#.cpu().squeeze(), y_true.cpu().squeeze())
    #     try:
    #         mevals.append(float(m.compute()))
    #     except ValueError:
    #         mevals.append(m.compute())
    acc = tm.Accuracy(task = "binary", average="micro") #,threshold=0) #should it not be 0.5?
    f1 = tm.F1Score(task='binary', num_classes=2, average="none")
    precision = tm.Precision(task = 'binary',num_classes=2, average="none")
    recall = tm.Recall(task='binary', num_classes=2, average="none")
    
    mevals.append(acc(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(f1(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(precision(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    mevals.append(recall(all_outputs.cpu().squeeze().round(),y_true.cpu().squeeze()))
    if return_vals:
        y_true = y_true.cpu().numpy().squeeze()
        #_, y_pred = # torch.max(all_outputs, 1) #for softmax
        y_pred = all_outputs.cpu().numpy().squeeze()
        return mevals, y_pred, y_true
    return mevals

def eval_multiclass(test_loader, model, device='cuda', metrics = [None], return_vals = False, n_classes=4):

    #metrics = [MulticlassAccuracy(num_classes = n_classes), MulticlassF1Score(num_classes = n_classes)]
    #(X,y, model, device = 'cuda', metrics = [MulticlassAccuracy]):
    #dataset = RegressionDataset(X, y)
    #alldata_loader = DataLoader(dataset = dataset, batch_size = 64, shuffle = False)
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    #metrics = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_true = torch.cat((y_true, targets), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    #if len(all_outputs[0]) > 1:
    #     print("example all_outputs[0]", all_outputs[0])
    #     print(torch.argmax(all_outputs[0]))#, torch.argmax(y_pred, dim=0).shape)
    #     y_pred = [torch.argmax(y) for y in y_pred] #axis=0? # [y.tolist() for y in y_pred]
    #     y_true = torch.Tensor(y_true).to(torch.long)
    # y_pred = torch.Tensor(y_pred).to(torch.long)
    mevals = []
    
    acc = tm.Accuracy(task='multiclass',average="none", num_classes = n_classes) #across all classes
    f1 = tm.F1Score(task='multiclass',num_classes=n_classes, average="none")
    precision = tm.Precision(task='multiclass',num_classes=n_classes, average="none")
    recall = tm.Recall(task='multiclass',num_classes=n_classes, average="none")
    
    y_pred = torch.argmax(all_outputs, dim= 1).cpu().round()

    mevals.append(acc(y_pred,y_true.cpu().squeeze()))
    mevals.append(f1(y_pred,y_true.cpu().squeeze()))
    mevals.append(precision(y_pred,y_true.cpu().squeeze()))
    mevals.append(recall(y_pred,y_true.cpu().squeeze()))
 
            
    if return_vals:
        y_true = y_true.cpu().numpy().squeeze()
        #_, y_pred = # torch.max(all_outputs, 1) #for softmax
        y_pred = y_pred.cpu().numpy().squeeze()
        return mevals, y_pred, y_true
    return mevals #, y_pred, y_true

