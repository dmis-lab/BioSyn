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

## NCBI Disease
You can preprocess NCBI disease dataset from scratch.
If you don't have the `NCBI-disease` dataset, you have to download it from the [website] (https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/).

First, parse the raw `NCBI-disease` data.
The result will be `mentions (*.concept)` and `contexts (*.txt)` 
```
DATA_DIR=../datasets

python ./ncbi_disease_preprocess.py \
    --input_file ${DATA_DIR}/raw/ncbi-disease/NCBItrainset_corpus.txt \
    --output_dir ${DATA_DIR}/ncbi-disease/train

python ./ncbi_disease_preprocess.py \
    --input_file ${DATA_DIR}/raw/ncbi-disease/NCBIdevelopset_corpus.txt \
    --output_dir ${DATA_DIR}/ncbi-disease/dev

python ./ncbi_disease_preprocess.py \
    --input_file ${DATA_DIR}/raw/ncbi-disease/NCBItestset_corpus.txt \
    --output_dir ${DATA_DIR}/ncbi-disease/test
```

Second, apply the text preprocess to the train/dev/test dataset and their dictionaries
```
DATA_DIR=../datasets
AB3P_PATH=../Ab3P/identify_abbr

# preprocess trainset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ./resources/medic_06Jul2012.txt \
    --output_dictionary_path ${DATA_DIR}/ncbi-disease/train_dictionary.txt \
    --lowercase \
    --remove_punctuation

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/ncbi-disease/train/ \
    --output_dir ${DATA_DIR}/ncbi-disease/processed_train/ \
    --dictionary_path ${DATA_DIR}/ncbi-disease/train_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --typo_path ./resources/ncbi-spell-check.txt \
    --remove_cuiless \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true

# preprocess devset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/ncbi-disease/train_dictionary.txt \
    --additional_data_dir ${DATA_DIR}/ncbi-disease/processed_train/ \
    --output_dictionary_path ${DATA_DIR}/ncbi-disease/dev_dictionary.txt \
    --lowercase \
    --remove_punctuation

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/ncbi-disease/dev/ \
    --output_dir ${DATA_DIR}/ncbi-disease/processed_dev/ \
    --dictionary_path ${DATA_DIR}/ncbi-disease/dev_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --typo_path ./resources/ncbi-spell-check.txt \
    --remove_cuiless \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true

# preprocess testset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/ncbi-disease/dev_dictionary.txt \
    --additional_data_dir ${DATA_DIR}/ncbi-disease/processed_dev \
    --output_dictionary_path ${DATA_DIR}/ncbi-disease/test_dictionary.txt \
    --lowercase \
    --remove_punctuation
    
python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/ncbi-disease/test/ \
    --output_dir ${DATA_DIR}/ncbi-disease/processed_test/ \
    --dictionary_path ${DATA_DIR}/ncbi-disease/test_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --typo_path ./resources/ncbi-spell-check.txt \
    --remove_cuiless \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true
```

## NLMChem
You can preprocess NLMChem dataset from scratch.

First, parse the raw `NLMChem` data.
The result will be `mentions (*.concept)` and `contexts (*.txt)` 
```
DATA_DIR=../datasets

python ./nlmchem_preprocess.py \
    --input_file ${DATA_DIR}/raw/NLMChem/BC7T2-NLMChem-corpus-train.BioC.json \
    --output_dir ${DATA_DIR}/NLMChem/train

python ./nlmchem_preprocess.py \
    --input_file ${DATA_DIR}/raw/NLMChem/BC7T2-NLMChem-corpus-dev.BioC.json \
    --output_dir ${DATA_DIR}/NLMChem/dev

python ./nlmchem_preprocess.py \
    --input_file ${DATA_DIR}/raw/NLMChem/BC7T2-NLMChem-corpus-test.BioC.json \
    --output_dir ${DATA_DIR}/NLMChem/test
```

Second, apply the text preprocess to the train/dev/test dataset and their dictionaries
```
DATA_DIR=../datasets
AB3P_PATH=../Ab3P/identify_abbr

# preprocess raw dictionary
python ctd_preprocess.py \
    --type chemical \
    --inpath /home/mujeen/works/bc7ner/chemical.tsv \
    --outpath /home/mujeen/works/BioSyn-dmis/preprocess/resources/ctd_chemical_30Jun2021.txt

# preprocess trainset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ./resources/ctd_chemical_30Jun2021.txt \
    --output_dictionary_path ${DATA_DIR}/NLMChem/train_dictionary.txt \
    --lowercase \
    --remove_punctuation

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/NLMChem/train/ \
    --output_dir ${DATA_DIR}/NLMChem/processed_train/ \
    --dictionary_path ${DATA_DIR}/NLMChem/train_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true \
    --filter_duplicate

# preprocess devset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/NLMChem/train_dictionary.txt \
    --additional_data_dir ${DATA_DIR}/NLMChem/processed_train/ \
    --output_dictionary_path ${DATA_DIR}/NLMChem/dev_dictionary.txt \
    --lowercase \
    --remove_punctuation

python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/NLMChem/dev/ \
    --output_dir ${DATA_DIR}/NLMChem/processed_dev/ \
    --dictionary_path ${DATA_DIR}/NLMChem/dev_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true \
    --filter_duplicate

# preprocess testset and its dictionary
python dictionary_preprocess.py \
    --input_dictionary_path ${DATA_DIR}/NLMChem/dev_dictionary.txt \
    --additional_data_dir ${DATA_DIR}/NLMChem/processed_dev \
    --output_dictionary_path ${DATA_DIR}/NLMChem/test_dictionary.txt \
    --lowercase \
    --remove_punctuation
    
python ./query_preprocess.py \
    --input_dir ${DATA_DIR}/NLMChem/test/ \
    --output_dir ${DATA_DIR}/NLMChem/processed_test/ \
    --dictionary_path ${DATA_DIR}/NLMChem/test_dictionary.txt \
    --ab3p_path ${AB3P_PATH} \
    --resolve_composites \
    --lowercase true \
    --remove_punctuation true \
    --filter_duplicate
```