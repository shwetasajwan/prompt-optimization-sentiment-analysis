# Prompt Optimization System for Sentiment Analysis Using Transformer-Based Models

## Overview

This project investigates whether different prompt engineering techniques
can influence the performance of traditional transformer-based models
for sentiment analysis.

The system compares multiple prompt styles across DistilBERT and RoBERTa
using the IMDB movie review dataset.

## Problem

Prompt engineering is widely used with large language models, but its
effect on traditional transformer-based models is less clear.

This project studies whether changing the way input text is presented
can affect sentiment classification performance.

## Prompt Techniques

The project evaluates four prompt styles:

1. Basic Prompt
2. Role-based Prompt
3. Structured Prompt
4. Few-shot Prompt

## Models

- DistilBERT
- RoBERTa

## Dataset

IMDB Movie Review Dataset

The dataset contains movie reviews labelled as positive or negative.

## Methodology

Dataset
↓
Data Preprocessing
↓
Prompt Generation
↓
Transformer Model
↓
Sentiment Prediction
↓
Performance Evaluation
↓
Model + Prompt Comparison

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

The experiments compare all prompt-model combinations.

The project report found that structured prompts produced relatively
consistent performance, while the choice of transformer model had a
larger effect on performance than prompt variation.

## Technologies

- Python
- Pandas
- Hugging Face Transformers
- DistilBERT
- RoBERTa
- Scikit-learn
- Matplotlib
- Seaborn



