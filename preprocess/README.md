# Pre-processing datasets and dictionaries

## Requirements

Install the `BioSyn` package first
```
cd .
python setup.py develop
```

Install the other dependenies
```
conda install pandas
```

## TAC2017ADR and MedDRA dictionary
If you have the `TAC2017ADR` dataset, you can pre-process the dataset.
Note that, you also need `MedDRA` dictionary (Please visit the [website](https://www.meddra.org/)). 


First, parse the raw `TAC2017ADR` data.
The result will be `mentions (*.concept)` and `contexts (*.txt)` 
```
DATA_DIR=../datasets

python ./tac2017adr_preprocess.py \
    --input_dir ${DATA_DIR}/raw/tac2017adr/train_xml \
    --output_dir ${DATA_DIR}/tac2017adr/train

python ./tac2017adr_preprocess.py \
    --input_dir ${DATA_DIR}/raw/tac2017adr/gold_xml \
    --output_dir ${DATA_DIR}/tac2017adr/test
```

Second, parse the `MedDRA v18.1` dictionary
```
DATA_DIR=../datasets
MEDDRA_DIR=${DATA_DIR}/dictionary/meddra18.1
python ./meddra_preprocess.py \
    --hlgt_path ${MEDDRA_DIR}/hlgt.asc \
    --hlt_path ${MEDDRA_DIR}/hlt.asc \
    --pt_path ${MEDDRA_DIR}/pt.asc \
    --llt_path ${MEDDRA_DIR}/llt.asc \
    --output_path ${DATA_DIR}/dictionary/meddra18.1.txt
```

Third, apply the text preprocess to the train/test dataset
```
DATA_DIR=../datasets
AB3P_PATH=../Ab3P/identify_abbr

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/tac2017adr/train/ \
    --output_dir ${DATA_DIR}/tac2017adr/processed_train/ \
    --ab3p_path ${AB3P_PATH} \
    --lowercase true \
    --remove_punctuation true

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/tac2017adr/test/ \
    --output_dir ${DATA_DIR}/tac2017adr/processed_test/ \
    --ab3p_path ${AB3P_PATH} \
    --lowercase true \
    --remove_punctuation true
```

Lastly, apply the text preprocess to the train/test dictionary.
Note that the only difference between the dictionaries is that test_dictionary includes train mentions to increase the coverage.
```
DATA_DIR=../datasets
python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/dictionary/meddra18.1.txt \
    --output_dictionary_path ${DATA_DIR}/tac2017adr/train_dictionary.txt \
    --lowercase true \
    --remove_punctuation true

python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/tac2017adr/train_dictionary.txt \
    --additional_data_dir ${DATA_DIR}/tac2017adr/processed_train \
    --output_dictionary_path ${DATA_DIR}/tac2017adr/test_dictionary.txt \
    --lowercase true \
    --remove_punctuation true
```
