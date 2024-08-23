import argparse
import sys
import pandas as pd
from datasets import Dataset, load_metric
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification, Dinov2ForImageClassification, Trainer, TrainingArguments, ResNetConfig, ResNetForImageClassification, Dinov2ForImageClassification, ViTMSNForImageClassification
import torch
from torchvision import transforms as tvtransforms
import numpy as np
from galaxy_mnist.galaxy_mnist import GalaxyMNIST
from fm_datasets import RGZimageDatasetClassification


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        #print("logits", logits.shape)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.Tensor([2.7068446 , 8.4972973 , 7.08108108, 8.57844475, 8.35059761, 7.38895417])).to('cuda')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="facebook/dinov2-base", type=str,help='')
    parser.add_argument('--output_dir', default=".", type=str,help='')
    parser.add_argument('--project', default="FM_compare", type=str,help='')
    parser.add_argument('--weights', default=None, type=str,help='')
    parser.add_argument('--ims_per_batch', default=8, type=int,help='')
    parser.add_argument('--nlabels', default=100, type=int,help='')
    parser.add_argument('--eid', default=1, type=int,help='experiment ID')
    parser.add_argument('--epochs', default=150, type=int,help='training epochs')
    parser.add_argument('--seed', default=15, type=int,help='')
    parser.add_argument('--thresh', default=0.5, type=float,help='')
    parser.add_argument('--lr', default=0.00005, type=float,help='')
    parser.add_argument('--resume', action = 'store_true',help='')
    parser.add_argument('--freeze_backbone', action = 'store_true',help='')
    parser.add_argument('--use_fp16', dest="use_fp16",action = 'store_true',help='')
    parser.set_defaults(use_fp16=False)
    parser.add_argument('--img_fmt', default='npy', type=str,help='')
    parser.add_argument('--dataset_train', default='/path/to/GMNIST') 

    #parser.add_argument('--cuda', default=['0'], nargs='+', help='')
    args = parser.parse_args()
    return args

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([x[1] for x in batch])
    }


def run_main(args):
    model_name_or_path = args.model_name
    if "resnet" in args.model_name:
        size = 256
    else:
        size = 224
    processor = ViTImageProcessor(do_normalize = False, size = size)
    stransforms = tvtransforms.Compose([tvtransforms.ToTensor(), tvtransforms.Resize([224,224])]) 
    
    if "MNIST" in args.dataset_train:
        train_ds = GalaxyMNIST(root=args.dataset_train, download = False, train=True, transform = stransforms)
        test_ds = GalaxyMNIST(root=args.dataset_train, download = False, train=False, transform = stransforms)
        labels = ["SR","SC","E","U"] 
    elif "rgz" in args.dataset_train:
        train_ds = RGZimageDatasetClassification(None, None, train=True, transform = stransforms)
        test_ds = RGZimageDatasetClassification(None, None, train=False, transform = stransforms)
        labels = ["1-1","1-2","1-3","2-2","2-3","3-3"]
    print(labels)
    if args.nlabels != 100: #get a subset - make sure to choose a seed which includes examples of all labels
        np.random.seed(args.seed)
        random_idx = np.random.choice(len(train_ds), size = args.nlabels, replace = False)
        train_ds = torch.utils.data.Subset(train_ds, random_idx)

    training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.ims_per_batch,
    evaluation_strategy="epoch",
    num_train_epochs=args.epochs,
    fp16=args.use_fp16,
    learning_rate=args.lr,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='wandb',
    metric_for_best_model="eval_loss",
    #load_best_model_at_end=True,
    )

    ametric = load_metric("accuracy", experiment_id = args.eid)
    pmetric = load_metric("precision", experiment_id = args.eid)
    rmetric = load_metric("recall", experiment_id = args.eid)
    fmetric = load_metric("f1", experiment_id = args.eid)
    
    def compute_metrics(p):
        acc = ametric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['accuracy']
        pre = pmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['precision']
        rec = rmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['recall']
        f1  = fmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['f1']
        return {"Accuracy":acc,"Precision":pre,"Recall":rec,"F1":f1}
    
    state_dict = None
    if args.weights:
        state_dict = torch.load(args.weights)["model_state_dict"]
    print(args.freeze_backbone,"state_dict", state_dict)
    if not "resnet" in args.model_name:
        if not args.freeze_backbone:
            # Initializing a ViT vit-base-patch16-224 style configuration. If backbone not frozen, this means training a ViT from scratch
            configuration = ViTConfig(num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)})
            model = ViTForImageClassification(configuration)
        elif "msn" in args.model_name:
            model = ViTMSNForImageClassification.from_pretrained(args.model_name,num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True, state_dict = state_dict)
        elif "dinov2" in args.model_name:
            model = Dinov2ForImageClassification.from_pretrained(args.model_name,num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True, state_dict = state_dict)
        else: #if backbone is frozen, pre-trained ViT will be loaded
            model = ViTForImageClassification.from_pretrained(args.model_name,num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True, state_dict = state_dict)
    else:
        if not args.freeze_backbone: #ResNet from scratch
            configuration = ResNetConfig(num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)})
            model = ResNetForImageClassification(configuration,)
        else: #ResNet from pretrained
            model = ResNetForImageClassification.from_pretrained(args.model_name,num_labels=len(labels),id2label={str(i): c for i, c in enumerate(labels)},label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True, state_dict = state_dict)
    
    if args.freeze_backbone: #do the actual freezing
        for name,p in model.named_parameters():
            if not name.startswith('classifier'):
                p.requires_grad = False

    if "MNIST" in args.dataset_train:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=processor,
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=processor,
        )

    train_results = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(test_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    args = get_args()
    run_main(args)