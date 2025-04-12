# RNN vs LSTM Experiment on Fake News Detection

This repository contains a comparative experiment to analyze the performance of RNN and LSTM models on fake news classification using Word2Vec embeddings. The dataset is sourced directly from KaggleHub: [hassanamin/textdb3](https://www.kaggle.com/datasets/hassanamin/textdb3).

## Features
- Dataset: `fake_or_real_news.csv` (from KaggleHub)
- Text cleaning and tokenization using `nltk`
- Word2Vec embedding generation with multiple dimensions (50, 100, 200)
- Sequence padding with varying lengths (50, 100, 200)
- RNN and LSTM architectures compared
- Classification metrics: Accuracy, Precision, Recall, F1-Score

## Files
- `rnn_lstm_experiment.py`: Main experiment script
- `requirements.txt`: Python package dependencies
- `rnn_lstm_experiment_results.csv`: Output of experiment (model performance)

## Installation
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python rnn_lstm_experiment.py
```

This will run all combinations of embedding dimensions, sequence lengths, model types (RNN vs LSTM), and hidden layer sizes, saving the evaluation metrics in a CSV file.

## Output
A CSV file (`rnn_lstm_experiment_results.csv`) will be generated containing the evaluation results:

| Model | Embed_Dim | Seq_Len | Hidden_Dim | Accuracy | Precision | Recall | F1-Score |
|-------|------------|----------|-------------|-----------|------------|--------|-----------|

## License
MIT License

---
Feel free to modify and extend this experiment for other NLP tasks or embedding strategies.
