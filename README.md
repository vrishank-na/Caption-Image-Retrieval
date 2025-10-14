# Caption-Image-Retrieval

## Overview
This machine learning project explores the task of image-caption retrieval using the COCO Common Objects in Context dataset, where each image is annotated with multiple human-written captions. The main goal is to align embeddings of captions and images such that, given a caption, the model retrieves the most semantically relevant image from the dataset.​

## Methods
#### Dataset: 
Preprocessed COCO dataset (Karpathy split) containing images and five captions per image; extracted VGG-19 image features are used for training.​

Our current implementation focuses on a smaller subset keeping feasability as a requirement , for better results larger datasets should be used.

#### Baselines:
Multi-response linear regression using averaged GloVe word embeddings for captions mapped to VGG-19 features.
Weighted baseline variant by averaging caption features per image.​

#### Neural Sequence Models:
GRU from scratch, GRU initialized with GloVe, LSTM with GloVe, and Bidirectional GRU with GloVe initialization.
Models encode captions using PyTorch with GloVe initialization for word embeddings, joint alignment to image features.​

Loss/Evaluation: Pairwise ranking loss is used, optimizing image retrieval metrics: Recall@1, Recall@10, and mean rank

## Results

#### Baseline Model results

| Model           | Implementation          | Recall@1 (%) | Recall@10 (%) | Mean Rank |
|-----------------|-------------------------|--------------|---------------|-----------|
| Baseline        | Original Paper          | 10.3         | 17.1          | 176.1     |
| Baseline        | Baseline_models.ipynb   | 36.2         | 90.4          | 4.7       |
| Baseline Weight | Original Paper          | 19.8         | 65.5          | 10.5      |
| Baseline Weight | Baseline_models.ipynb   | 44.0         | 94.0          | 3.7       |


#### Neural Sequence Model Results

| Model            | Implementation            | Recall@1 (%) | Recall@10 (%) | Mean Rank |
|------------------|---------------------------|--------------|---------------|-----------|
| GRU from scratch | seq_neural_models.ipynb   | 0.1          | 0.6           | 508.1     |
| GRU + GloVe      | Original Paper            | 37.0         | 86.8          | 7.3       |
| GRU + GloVe      | seq_neural_models.ipynb   | 0.2          | 0.7           | 504.3     |
| LSTM + GloVe     | Original Paper            | 35.4         | 86.2          | 7.5       |
| LSTM + GloVe     | seq_neural_models.ipynb   | 0.2          | 0.7           | 512.7     |
| BiGRU + GloVe    | seq_neural_models.ipynb   | 0.1          | 0.6           | 509.4     |


## How to Run
1. Baseline_models.ipynb: The baseline and the baseline weight models are trained in this ipynb file.

2. seq_neural_models.ipynb: The implementations of the neural models is done in this ipynb file.

The datasets in both the ipynb files are inbuilt.

## Resources:

Reference Paper: https://cs229.stanford.edu/proj2018/report/59.pdf

This project was inspired by or references the original implementation in [Original Repo] ((https://github.com/giacomolamberti90/CS229_project)).  




