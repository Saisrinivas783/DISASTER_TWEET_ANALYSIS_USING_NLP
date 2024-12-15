# Disaster Tweet Analysis using NLP

## Project Overview

This project implements a Natural Language Processing (NLP) pipeline to analyze disaster-related tweets. It utilizes AWS services for data storage and processing, and employs advanced NLP techniques including BERT-based classification.

## Key Components

- Data Storage: Amazon S3
- Data Processing: AWS Glue (implied from S3 usage)
- Machine Learning: BERT model implemented with PyTorch and Hugging Face Transformers

## Features

- Comprehensive data preprocessing
- Word frequency analysis
- N-gram analysis (unigrams, bigrams, trigrams)
- BERT-based binary classification (disaster vs. non-disaster tweets)

## Technical Stack

- Python
- pandas, numpy, matplotlib, seaborn
- NLTK, spaCy
- PyTorch, Transformers (Hugging Face)
- AWS SDK (boto3)

## Data Preprocessing

The preprocessing pipeline includes:

- Tokenization
- Lowercasing
- Stopword removal
- Part-of-speech tagging
- Lemmatization

## Model Architecture

- Uses `bert-base-uncased` pretrained model
- Fine-tuned for binary classification
- Implemented using PyTorch and Hugging Face Transformers

## Training Process

- Dataset split: 80% training, 20% validation
- Batch size: 32
- Learning rate: 6e-6
- Number of epochs: 3
- Uses AdamW optimizer and linear learning rate scheduler

## Visualization

The project includes various visualizations:

- Word frequency plots for disaster and non-disaster tweets
- N-gram analysis charts (unigrams, bigrams, trigrams)
- Learning curve plots for training and validation loss

## Results

The model's performance is evaluated using accuracy and F1 score. Detailed training statistics are provided, including training/validation loss and accuracy for each epoch.

## Model Performance

The BERT-based classification model for disaster tweet analysis showed promising results over three epochs of training:

### Training and Validation Metrics

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation F1 Score |
|-------|---------------|-----------------|---------------------|---------------------|
| 1     | 0.50          | 0.44            | 0.82                | 0.78                |
| 2     | 0.39          | 0.42            | 0.82                | 0.77                |
| 3     | 0.36          | 0.42            | 0.83                | 0.78                |

### Key Observations

- **Training Loss**: Decreased consistently from 0.50 to 0.36 across the three epochs, indicating good learning progress.

- **Validation Loss**: Stabilized around 0.42-0.44, suggesting the model generalizes well to unseen data.

- **Validation Accuracy**: Improved slightly from 0.82 to 0.83, demonstrating consistent performance.

- **Validation F1 Score**: Remained stable at 0.78, indicating a good balance between precision and recall.

The model achieved strong performance with an accuracy of 83% and an F1 score of 0.78 on the validation set by the final epoch. The consistent validation metrics across epochs suggest the model is not overfitting and has learned to generalize well to new disaster-related tweets.
