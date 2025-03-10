import argparse
import os
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")

class ReportSummarizer:
    def __init__(self, method='extractive', model_name=None, max_length=150, min_length=40):
        """
        Initialize the report summarizer.
        
        Args:
            method (str): Summarization method ('extractive' or 'abstractive')
            model_name (str, optional): Pretrained model name for abstractive summarization
            max_length (int): Maximum length of the generated summary
            min_length (int): Minimum length of the generated summary
        """
        self.method = method
        self.max_length = max_length
        self.min_length = min_length
        
        if self.method == 'abstractive':
            if not model_name:
                model_name = 'facebook/bart-large-cnn'
            
            try:
                logger.info(f"Loading abstractive model: {model_name}")
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
                self.model = BartForConditionalGeneration.from_pretrained(model_name)
                self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            except Exception as e:
                logger.error(f"Failed to load abstractive model: {e}")
                logger.info("Falling back to extractive summarization")
                self.method = 'extractive'
        
        if self.method == 'extractive':
            logger.info("Using extractive summarization")
            self.stop_words = set(stopwords.words('english'))
            self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def preprocess_text(self, text):
        """
        Preprocess text by cleaning and tokenizing.
        
        Args:
            text (str): Input text to be preprocessed
            
        Returns:
            list: List of sentences
        """
        # Basic cleaning
        text = text.strip()
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        return sentences
    
    def extractive_summarize(self, text):
        """
        Generate an extractive summary using TF-IDF ranking.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            str: Extractive summary
        """
        sentences = self.preprocess_text(text)
        
        if len(sentences) <= 3:
            return text
        
        # Create sentence vectors
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores based on TF-IDF values
        sentence_scores = np.array([tfidf_matrix[i].sum() for i in range(len(sentences))])
        
        # Determine number of sentences for summary (approximately 1/3 of original)
        num_sentences = max(min(int(len(sentences) / 3), 10), 3)
        
        # Get indices of top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:]
        top_indices = sorted(top_indices)
        
        # Construct summary
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def abstractive_summarize(self, text):
        """
        Generate an abstractive summary using pretrained transformer model.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            str: Abstractive summary
        """
        try:
            summary = self.summarizer(text, 
                                      max_length=self.max_length, 
                                      min_length=self.min_length, 
                                      do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            logger.info("Falling back to extractive summarization")
            return self.extractive_summarize(text)
    
    def summarize(self, text):
        """
        Generate a summary based on the selected method.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            str: Generated summary
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        if self.method == 'abstractive':
            return self.abstractive_summarize(text)
        else:
            return self.extractive_summarize(text)
    
    def summarize_file(self, file_path):
        """
        Read a file and generate a summary.
        
        Args:
            file_path (str): Path to the input file
            
        Returns:
            str: Generated summary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.summarize(text)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return ""
    
    def summarize_directory(self, dir_path, output_dir=None, file_ext='.txt'):
        """
        Summarize all files in a directory with the specified extension.
        
        Args:
            dir_path (str): Path to the directory containing files
            output_dir (str, optional): Directory to save summaries
            file_ext (str): File extension to filter (default: .txt)
            
        Returns:
            dict: Dictionary mapping filenames to summaries
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(file_ext):
                file_path = os.path.join(dir_path, filename)
                logger.info(f"Summarizing: {filename}")
                
                summary = self.summarize_file(file_path)
                results[filename] = summary
                
                if output_dir:
                    output_file = os.path.join(output_dir, f"summary_{filename}")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(summary)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Report Summarizer Tool')
    parser.add_argument('--input', required=True, help='Input file or directory path')
    parser.add_argument('--output', help='Output directory path (for directory processing)')
    parser.add_argument('--method', choices=['extractive', 'abstractive'], default='extractive', 
                        help='Summarization method (extractive or abstractive)')
    parser.add_argument('--model', default=None, help='Model name for abstractive summarization')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum summary length')
    parser.add_argument('--min_length', type=int, default=40, help='Minimum summary length')
    parser.add_argument('--file_ext', default='.txt', help='File extension for directory processing')
    
    args = parser.parse_args()
    
    summarizer = ReportSummarizer(
        method=args.method,
        model_name=args.model,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
        output_dir = args.output if args.output else os.path.join(args.input, 'summaries')
        summarizer.summarize_directory(args.input, output_dir, args.file_ext)
        logger.info(f"Summaries saved to: {output_dir}")
    else:
        logger.info(f"Processing file: {args.input}")
        summary = summarizer.summarize_file(args.input)
        
        if args.output:
            if os.path.isdir(args.output):
                output_file = os.path.join(args.output, f"summary_{os.path.basename(args.input)}")
            else:
                output_file = args.output
                
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Summary saved to: {output_file}")
        else:
            print("\nSUMMARY:\n========\n")
            print(summary)

if __name__ == "__main__":
    main()
