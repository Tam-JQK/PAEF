# Permutation-aware Exposure Fairness via Bi-level Optimization for Provider in Recommender Systems

**PAEF** 

## Dataset Description
As summarized in the following table, we select three public datasets, i.e., **Amazon_Baby**, **Book-Crossing** and **BeerAdvocate** to conduct experiments considering typical domains (books, games and beverages). 

| Datasets | #Users  | #Items  | #Interactions | # Providers |
| -------  | ------- | ------- | ------------- | ----------- |
| Amazon_Baby      |    15,301  | 5,824   | 124,436    | 646   |
| Book-Crossing |   6,400  | 8,496  | 104,918    | 607 |
| BeerAdvocate       |   14,040  | 20,561  | 1361,000  | 2,694 |

## Quick Start

1. Unzip dataset files.
    ```bash
    cd dataset/Amazon_Baby/; unzip Amazon_Baby.zip
    ```
2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3. Evaluate the performance of LightGCN model with our approach on Amazon_Baby dataset.
    ```bash
    python run_model.py --model=PARF_LightGCN --dataset=Amazon_Baby
    ```

## Parameter Configurations

### Configuration files

Parameter configurations consist of three parts/files: 
- overall configuration ([[overall.yaml]](props/overall.yaml))
- dataset configuration (e.g., [[Amazon_Baby.yaml]](props/Amazon_Baby.yaml))
- model configuration (e.g., [[PAEF_LightGCN.yaml]](props/PAEF_LightGCN.yaml))

### General configurations

To ensure a fair comparison, we optimize all the models with _Adam optimizer_ and parameters are initialized by the default _Xavier distribution_. 

The training batch size is set to 512 and the embedding size is set to 64. We stop training when the performance on the validation set does not improve for 20 epochs, and NDCG@20 is set as the indicator. 

## Acknowledgement
- The experiments are conducted using the open-source recommendation library RecBole.
