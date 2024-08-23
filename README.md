# fm4astro

Code for the paper submitted to **Vision foundation models: can they be applied to astrophysics data?** submitted to *Foundation Models for Science: Progress, Opportunities, and Challenges* workshop for NeurIPS 2024.

## Dependencies
- PyTorch, torchvision, huggingface transformers
- slurm
- [Faster-RCNN](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline) for object detection
- [mgcls_dino](https://github.com/elastufka/mgcls_dino) for MGCLS data preparation and ResNet feature extraction
- [pyBDSF_to_COCO](https://github.com/elastufka/pyBDSF_to_COCO) for preparing COCO labels for MGCLS

## Data
- [GalaxyMNIST]()
- Radio Galaxy Zoo
- [MGCLS]()

## Usage

### Feature extraction from foundation models

**ViT models**

via SLURM: see sbatch_scripts/extract_feats_gmnist.sh and extract_feats_rgz.sh

via command line (slow with only CPU):

```bash
python extract_features.py --dataset_train $data_path --img_fmt PIL --model facebook/vit-mae-base
mv features.pth "$dname"_trainfeat_maeB.pth

python extract_features.py --dataset_train $data_path --test --img_fmt PIL --model facebook/vit-mae-base
mv features.pth "$dname"_testfeat_maeB.pth
```

If output shape of features is wrong, run get_CLS.py:

```bash
python get_CLS.py --file_name "$dname"_trainfeat_maeB.pth
```

**ResNet models**

modify mgcls_dino/eval_knn_train.py in the following ways:
- add to initial if statement (along with imports for GalaxyMNIST and fm_datasets):
```python
    elif "rgz" in args.data_path:
        dataset_train = ReturnIndexDatasetRGZod(None, None, train=args.train, transform=transform)
    elif "GalaxyMNIST" in args.data_path:
        dataset_train = GalaxyMNIST(root=args.data_path, train=args.train, transform=transform)
```
- edit line 100 to read:  
```python
model = torchvision_models.__dict__[args.arch](weights='DEFAULT')
```
- comment out line 112 
```python 
#utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
```

via SLURM: see sbatch_scripts/extract_feats_resnet.sh

via command line (slow with only CPU):

```bash
data_path=/path/to/rgz
dump_path=/path/to/output/RGZ

python eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1  --in_chans 3 --resize 256 --center_crop 256
mv $dump_path/trainfeat.pth $dump_path/RGZ_trainfeat_RN50.pth

python eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1 --train false  --in_chans 3 --center_crop 256 --resize 256
mv $dump_path/trainfeat.pth $dump_path/RGZ_testfeat_RN50.pth
```

If output shape of features is wrong, run average_features.py:

```bash
python average_features.py --file_name "$dname"_trainfeat_RN50.pth
```

### Training foundation models from scratch

via SLURM: see sbatch_scripts/train_gmnist_from_scratch.sh and train_rgz_from_scratch.sh

via command line (slow with only CPU):

```bash
# train ViT-Base 16x16 from scratch with 30% of GMNIST training labels
python  finetune.py --epochs 100 --output_dir "GMNIST/ViTB/30p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 2400 --lr 0.000005
```

### Classification on frozen features

Update config.yaml as necessary for evaluating different models.

```bash
python  downstream.py --config eval_gmnist.yaml
```

### Object detection

Create compatible .yaml files that describe the datasets to be used.

Use model weights from /weights directory, which are compatible with fastercnn-pytorch-training-pipeline (only needed for ViT models).

Move train_frcnn.py into the top level of the fastercnn-pytorch-training-pipeline directory. 

For using resnet18, move into fastercnn-pytorch-training-pipeline/models and update the __init__.py and any other code necessary.

via SLURM: see sbatch_scripts/$dataset_name_frcnn_$model.sh

via command line (slow with only CPU):

```bash
data=/path/to/rgz/rcnn_dataset_30p.yaml
python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_finetune_30p --batch 16 -uta --mosaic 0 -ims 256 
```

