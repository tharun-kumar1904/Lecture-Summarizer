import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import defaultdict
import string
import math
import torch
try:
    from transformers import BartForConditionalGeneration, BartTokenizer
except ImportError:
    BartForConditionalGeneration = None
    BartTokenizer = None

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextSummarizer:
    def __init__(self):
        print("Initializing TextSummarizer...")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_bart = False
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        try:
            if BartForConditionalGeneration is not None and BartTokenizer is not None:
                print("Loading DistilBART model...")
                self.tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')
                self.model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6').to(self.device)
                self.use_bart = True
                print("DistilBART model loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize DistilBART model: {e}. Using TextRank fallback.")

    def preprocess_text(self, text):
        """Preprocess text for TextRank: tokenize, remove stopwords, and lemmatize."""
        sentences = sent_tokenize(text)
        processed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [self.lemmatizer.lemmatize(word) for word in words 
                    if word not in self.stop_words and word not in string.punctuation]
            processed_sentences.append(words)
        return sentences, processed_sentences

    def build_similarity_matrix(self, sentences):
        """Create a similarity matrix using cosine similarity for TextRank."""
        n = len(sentences)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = self.cosine_similarity(sentences[i], sentences[j])
        return sim_matrix

    def cosine_similarity(self, sent1, sent2):
        """Calculate cosine similarity between two sentences."""
        word_set = set(sent1).union(set(sent2))
        freq1 = defaultdict(int)
        freq2 = defaultdict(int)
        for word in sent1:
            freq1[word] += 1
        for word in sent2:
            freq2[word] += 1
        vector1 = [freq1[word] for word in word_set]
        vector2 = [freq2[word] for word in word_set]
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = np.sqrt(sum(a * a for a in vector1))
        norm2 = np.sqrt(sum(b * b for b in vector2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def textrank(self, sim_matrix, max_iter=100, d=0.85):
        """Implement TextRank algorithm for sentence scoring."""
        n = len(sim_matrix)
        scores = np.ones(n) / n
        for _ in range(max_iter):
            prev_scores = scores.copy()
            for i in range(n):
                scores[i] = (1 - d) + d * sum(sim_matrix[j][i] * prev_scores[j] / 
                                            sum(sim_matrix[j]) for j in range(n) if sum(sim_matrix[j]) != 0)
            if np.allclose(scores, prev_scores, rtol=1e-5):
                break
        return scores

    def summarize_textrank(self, text, ratio=0.3, progress_callback=None):
        """Generate extractive summary using TextRank, targeting input_words * ratio."""
        if not text.strip():
            return "Please enter some text to summarize."
        if progress_callback:
            progress_callback(40)
        original_sentences, processed_sentences = self.preprocess_text(text)
        if not original_sentences:
            return "No valid sentences found in the input text."
        input_words = len(text.split())
        target_words = math.ceil(input_words * ratio)
        print(f"TextRank: Input {input_words} words, Target {target_words} words")
        if progress_callback:
            progress_callback(50)
        sim_matrix = self.build_similarity_matrix(processed_sentences)
        if progress_callback:
            progress_callback(60)
        scores = self.textrank(sim_matrix)
        ranked_sentences = [(score, sent, len(sent.split())) for score, sent in zip(scores, original_sentences)]
        ranked_sentences.sort(reverse=True)
        summary_sentences = []
        current_words = 0
        for _, sent, word_count in ranked_sentences:
            if current_words < target_words:
                summary_sentences.append(sent)
                current_words += word_count
            else:
                break
        summary = []
        for sent in original_sentences:
            if sent in summary_sentences:
                summary.append(sent)
        if progress_callback:
            progress_callback(70)
        summary_text = " ".join(summary)
        actual_words = len(summary_text.split())
        print(f"TextRank: Actual {actual_words} words")
        return summary_text

    def summarize_bart(self, text, ratio=0.3, progress_callback=None):
        """Generate abstractive summary using DistilBART, targeting input_words * ratio."""
        if not text.strip():
            return "Please enter some text to summarize."
        if progress_callback:
            progress_callback(40)
        input_words = len(text.split())
        target_words = math.ceil(input_words * ratio)
        min_length = target_words
        max_length = int(target_words * 1.5)  # Reduced for speed
        est_tokens = int(target_words * 1.2)
        print(f"BART: Input {input_words} words, Target {target_words} words, min_length {min_length}, max_length {max_length}, length_penalty 3.0, num_beams 4, repetition_penalty 2.0, no_repeat_ngram_size 3, est_tokens {est_tokens}")
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        input_tokens = len(inputs['input_ids'][0])
        if progress_callback:
            progress_callback(50)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            length_penalty=3.0,  # Balanced for speed
            num_beams=4,  # Reduced for speed
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        if progress_callback:
            progress_callback(70)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        actual_words = len(summary.split())
        output_tokens = len(summary_ids[0])
        token_ratio = output_tokens / actual_words if actual_words > 0 else 0
        print(f"BART: Actual {actual_words} words, Input tokens {input_tokens}, Output tokens {output_tokens}, Token/Word ratio {token_ratio:.2f}")
        if actual_words < target_words * 0.9:  # Higher threshold for speed
            print(f"BART output too short ({actual_words}/{target_words} words). Triggering TextRank fallback.")
            return self.summarize_textrank(text, ratio, progress_callback)
        return summary

    def summarize(self, text, ratio=0.3, use_bart=True, progress_callback=None):
        """Generate summary, preferring BART if available and selected."""
        if use_bart and self.use_bart:
            try:
                return self.summarize_bart(text, ratio, progress_callback)
            except Exception as e:
                print(f"BART summarization failed: {e}. Using TextRank fallback.")
                return self.summarize_textrank(text, ratio, progress_callback)
        return self.summarize_textrank(text, ratio, progress_callback)