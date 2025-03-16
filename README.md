# MM-CTR-Multimodal-CTR-Prediction-Challenge
## Solution
### 1.Qwen2.5-vl-7B to get emb
using the second-to-last hidden layer with mean pooling to obtain embeddings
### 2.Using PCA to reduce the dimensionality of embeddings
### 3.Using the default settings of the DIN recommendation model
Hyperparameter scheme:  
embedding_regularizer: 0  
net_regularizer: 0  
net_dropout: 0.11  
learning_rate: 2.e-3  
batch_size: 7168  
## Run
### Data Preparation

1. Download the datasets at: https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

2. Unzip the data files to the `data` directory

```bash
cd ~/mmctr-solusion-by-delorean/data/
find -L .

.
./MicroLens_1M_x1
./MicroLens_1M_x1/train.parquet
./MicroLens_1M_x1/valid.parquet
./MicroLens_1M_x1/test.parquet
./MicroLens_1M_x1/item_info.parquet
./item_feature.parquet
./item_emb.parquet   
./item_seq.parquet  
./item_images.rar  
```
### Model Download
Because the size of LLM and the bad internet connection, please download model folder and put it at main dir  
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
### Run code
1.Running all code  
```./run.sh```  
2.Only part1: get embeddings and PCA  
```python inference.py```  
```python PCA.py```  
3.Only part2: train DIN and predict  
```python gen_item_info.py```  
```python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_01.yaml --gpu 0```  
```python prediction.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x1_xxx --gpu 0```  

