<h3 align="center">
<p>BioSyn
<a href="https://github.com/dmis-lab/BioSyn/blob/master/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
</h3>
<div align="center">
    <p><b>Bio</b>medical entity representations with <b>Syn</b>onym marginalization
</div>

<div align="center">
  <img alt="BioSyn Overview" src="https://github.com/dmis-lab/biosyn/blob/master/images/biosyn_overview.png" width="500px">
</div>

We present BioSyn for learning and normalizing entity representations in the biomedical domain, built upon two main components: 1) synonym marginalization and 2) iterative candidate retrieval.

## Requirements

Note: Pytorch has to be installed depending on the version of CUDA.

```
$ conda create -n BioSyn python=3.6
$ conda activate BioSyn
$ conda install numpy
$ conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
$ conda install tqdm
$ conda install nltk
$ conda install scikit-learn
$ pip install transformers
```

## Resources

### Pretrained Model
[BioBERTv1.1](https://github.com/dmis-lab/biobert) of the
[Huggingface](https://github.com/huggingface/transformers) version so that the pretrained model can be run on the pytorch framework.

- [biobertv1.1 (pytorch)](https://drive.google.com/drive/folders/1nSjj-ubecQbwYPdz3NyAqiJ1-rLtguUp?usp=sharing)

### Datasets

Datasets consist of queries (train, dev, test, and traindev), and dictionaries (train_dictionary, dev_dictionary, and test_dictionary). The queries are pre-processed with lowercasing, removing punctuations, resolving composite mentions and resolving abbreviation ([Ab3P](https://github.com/ncbi-nlp/Ab3P)). The dictionaries are pre-processed with lowercasing, removing punctuations. (If you need the pre-processing codes, please let me know by openning an issue.)

Note that we use development (dev) set to search the hyperparameters, and train on traindev (train+dev) set to report the final performance.

- [ncbi-disease](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj)
- [bc5cdr-disease](https://drive.google.com/open?id=1nvNYdfGrlZjya4RlhRu-IQJjRJzQcpyr)
- [bc5cdr-chemical](https://drive.google.com/open?id=1nsWIWmds5p7UZIeqrKVnhNTaBQAbqVYk)

('TAC2017ADR' dataset cannot be shared because of the license issue. Please visit the [website](https://bionlp.nlm.nih.gov/tac2017adversereactions/).)

## Train

The following example is fine-tuning our model with NCBI-Disease dataset (train+dev) on BioBERTv1.1. 

```
MODEL=biosyn-ncbi-disease
BIOBERT_DIR=./pretrained/pt_biobert1.1
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python ./train.py \
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

Note that you can train the model on 'processed_train' and evaluate it on 'processed_dev' when you want to search for the hyperparameters. (the argument '--save_checkpoint_all' can be helpful. )

## Evaluation

The following example is evaluating our trained model with NCBI-Disease dataset (test). 

```
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

The predictions are saved in 'predictions_eval.json' with mentions, candidates and accuracies. (the argument '--save_predictions' has to be on)
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
    }],
    "acc1": 0.9114583333333334,
    "acc5": 0.9385416666666667
}
```

## Citations

```
@misc{sung2020biomedical,
    title={Biomedical Entity Representations with Synonym Marginalization},
    author={Mujeen Sung and Hwisang Jeon and Jinhyuk Lee and Jaewoo Kang},
    year={2020},
    eprint={2005.00239},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
