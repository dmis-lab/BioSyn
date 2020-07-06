<h3 align="center">
<p>BioSyn
<a href="https://github.com/dmis-lab/BioSyn/blob/master/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
</h3>
<div align="center">
    <p><b>Bio</b>medical Entity Representations with <b>Syn</b>onym Marginalization
</div>

<div align="center">
  <img alt="BioSyn Overview" src="https://github.com/dmis-lab/biosyn/blob/master/images/biosyn_overview.png" width="500px">
</div>

We present BioSyn for learning biomedical entity representations. You can train BioSyn with the two main components described in our [paper](https://arxiv.org/abs/2005.00239): 1) synonym marginalization and 2) iterative candidate retrieval. Once you train BioSyn, you can easily normalize any biomedical mentions or represent them into entity embeddings. You can try our running demo [http://biosyn.korea.ac.kr](http://biosyn.korea.ac.kr) for nomalization of disease named entities.

## Requirements
```bash
$ conda create -n BioSyn python=3.6
$ conda activate BioSyn
$ conda install numpy tqdm nltk scikit-learn
$ conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
$ pip install transformers==2.0.0
```
Note that Pytorch has to be installed depending on the version of CUDA.

## Resources

### Pretrained Model
We use the [Huggingface](https://github.com/huggingface/transformers) version of [BioBERT v1.1](https://github.com/dmis-lab/biobert) so that the pretrained model can be run on the pytorch framework.

- [biobert v1.1 (pytorch)](https://drive.google.com/drive/folders/1nSjj-ubecQbwYPdz3NyAqiJ1-rLtguUp?usp=sharing)

### Datasets

Datasets consist of queries (train, dev, test, and traindev), and dictionaries (train_dictionary, dev_dictionary, and test_dictionary). Note that the only difference between the dictionaries is that test_dictionary includes train and dev mentions, and dev_dictionary includes train mentions to increase the coverage. The queries are pre-processed with lowercasing, removing punctuations, resolving composite mentions and resolving abbreviation ([Ab3P](https://github.com/ncbi-nlp/Ab3P)). The dictionaries are pre-processed with lowercasing, removing punctuations (If you need the pre-processing codes, please let us know by openning an issue).

Note that we use development (dev) set to search the hyperparameters, and train on traindev (train+dev) set to report the final performance.

- [ncbi-disease](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj)
- [bc5cdr-disease](https://drive.google.com/open?id=1nvNYdfGrlZjya4RlhRu-IQJjRJzQcpyr)
- [bc5cdr-chemical](https://drive.google.com/open?id=1nsWIWmds5p7UZIeqrKVnhNTaBQAbqVYk)

`TAC2017ADR` dataset cannot be shared because of the license issue. Please visit the [website](https://bionlp.nlm.nih.gov/tac2017adversereactions/) or see [here](https://github.com/dmis-lab/BioSyn/tree/master/preprocess) for pre-processing scripts.

## Train

The following example fine-tunes our model on NCBI-Disease dataset (train+dev) with BioBERTv1.1. 

```bash
MODEL=biosyn-ncbi-disease
BIOBERT_DIR=./pretrained/pt_biobert1.1
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python train.py \
    --model_dir ${BIOBERT_DIR} \
    --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --train_dir ${DATA_DIR}/processed_traindev \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 16\
    --initial_sparse_weight 0\
    --learning_rate 1e-5 \
    --max_length 25 \
    --dense_ratio 0.5
```

Note that you can train the model on `processed_train` and evaluate it on `processed_dev` when you want to search for the hyperparameters. (the argument `--save_checkpoint_all` can be helpful. )

## Evaluation

The following example evaluates our trained model with NCBI-Disease dataset (test). 

```bash
MODEL=biosyn-ncbi-disease
MODEL_DIR=./tmp/${MODEL}
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python eval.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions
```

### Result

The predictions are saved in `predictions_eval.json` with mentions, candidates and accuracies (the argument `--save_predictions` has to be on).
Following is an example.

```
{
  "queries": [
    {
      "mentions": [
        {
          "mention": "ataxia telangiectasia",
          "golden_cui": "D001260",
          "candidates": [
            {
              "name": "ataxia telangiectasia",
              "cui": "D001260|208900",
              "label": 1
            },
            {
              "name": "ataxia telangiectasia syndrome",
              "cui": "D001260|208900",
              "label": 1
            },
            {
              "name": "ataxia telangiectasia variant",
              "cui": "C566865",
              "label": 0
            },
            {
              "name": "syndrome ataxia telangiectasia",
              "cui": "D001260|208900",
              "label": 1
            },
            {
              "name": "telangiectasia",
              "cui": "D013684",
              "label": 0
            }]
        }]
    },
    ...
    ],
    "acc1": 0.9114583333333334,
    "acc5": 0.9385416666666667
}
```

## Inference
We provide a simple script that can normalize a biomedical mention or represent the mention into an embedding vector with BioSyn. If you do not have pre-trained BioSyn, please download [BioSyn pre-trained on NCBI-Disease](https://drive.google.com/drive/folders/1oOkY1Vtn508i0Q542IcJKFVm_d40Xj1I?usp=sharing).

### Predictions (Top 5)

The example below gives the top 5 predictions for a mention `ataxia telangiectasia`. Note that the initial run will take some time to embed the whole dictionary. You can download the dictionary file [here](https://github.com/dmis-lab/BioSyn#datasets).

```bash
MODEL=biosyn-ncbi-disease
MODEL_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python inference.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --use_cuda \
    --mention "ataxia telangiectasia" \
    --show_predictions
```

#### Result
```json
{
  "mention": "ataxia telangiectasia", 
  "predictions": [
    {"name": "ataxia telangiectasia", "id": "D001260|208900"}, 
    {"name": "ataxia telangiectasia syndrome", "id": "D001260|208900"}, 
    {"name": "ataxia telangiectasia variant", "id": "C566865"}, 
    {"name": "syndrome ataxia telangiectasia", "id": "D001260|208900"}, 
    {"name": "telangiectasia", "id": "D013684"}
  ]
}
```

### Embeddings
The example below gives an embedding of a mention `ataxia telangiectasia`.

```bash
MODEL=biosyn-ncbi-disease
MODEL_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python inference.py \
    --model_dir ${MODEL_DIR} \
    --use_cuda \
    --mention "ataxia telangiectasia" \
    --show_embeddings
```

#### Result
```
{
  "mention": "ataxia telangiectasia", 
  "mention_sparse_embeds": array([0.05979538, 0., ..., 0., 0.], dtype=float32), 
  "mention_dense_embeds": array([-7.14258850e-02, ..., -4.03847933e-01,],dtype=float32)
}
```

## Demo

### How to run web demo

Web demo is implemented on [Tornado](https://www.tornadoweb.org/) framework.
If a dictionary is not yet cached, it will take about couple of minutes to create dictionary cache.

```bash
MODEL=biosyn-ncbi-disease
MODEL_DIR=./tmp/${MODEL}

python demo.py \
  --model_dir ${MODEL_DIR} \
  --use_cuda \
  --dictionary_path ./datasets/ncbi-disease/test_dictionary.txt
```

You can try our running [demo](http://biosyn.korea.ac.kr) for nomalization of disease named entities.


## Citations
```bibtex
@inproceedings{sung2020biomedical,
    title={Biomedical Entity Representations with Synonym Marginalization},
    author={Sung, Mujeen and Jeon, Hwisang and Lee, Jinhyuk and Kang, Jaewoo},
    booktitle={ACL},
    year={2020},
}
```
