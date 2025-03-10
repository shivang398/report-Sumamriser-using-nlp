# report-Sumamriser-using-nlp
# Report Summarizer

## Overview
The **Report Summarizer** is a Python tool that provides both **extractive** and **abstractive** summarization for text-based reports. It processes individual files or directories containing text files and generates concise summaries.

## Features
- **Extractive Summarization:** Uses TF-IDF scoring to extract key sentences.
- **Abstractive Summarization:** Uses a pre-trained transformer model (BART) for generating summaries.
- **Batch Processing:** Can summarize multiple files within a directory.
- **Logging & Error Handling:** Provides logging for errors and status updates.

## Dependencies
Ensure you have the following libraries installed before running the script:
```bash
pip install numpy pandas nltk scikit-learn transformers torch

## Installation

Clone the repository and install dependencies:


git clone https://github.com/your-repo/report-summarizer.git
cd report-summarizer
pip install -r requirements.txt



