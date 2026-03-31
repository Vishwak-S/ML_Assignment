# Comparative Analysis of Word Vectorization Techniques for Sentiment Classification

**Course:** ML for Cyber Security Assignment 2026  
**Dataset:** IMDB 50K Movie Reviews Dataset  
**Team Members:** `Vishwak S`, `Girisudhan K`  

## Abstract
This work analyzes how different text vectorization techniques affect sentiment classification performance on the IMDB movie review dataset. Since machine learning models cannot directly process raw text, it is necessary to transform text into numerical vectors. In this assignment, three vectorization approaches were used: Bag of Words (BoW), TF-IDF, and GloVe embeddings. For classification, Logistic Regression and Linear Support Vector Machine (Linear SVM) were applied to each representation. The experiments show that TF-IDF with Logistic Regression achieved the best overall performance with an accuracy of 88.65%, while GloVe with simple mean pooling produced lower accuracy but denser semantic representations. The results indicate that conventional sparse methods remain highly competitive for sentiment analysis when combined with strong linear classifiers.

## 1. Introduction
Natural Language Processing (NLP) deals with textual data, but machine learning algorithms require numerical input. Word vectorization is the process of converting text into numerical form so that models can learn from linguistic patterns. The quality of this representation has a direct influence on the final classification accuracy, training speed, and model generalization.

Sentiment analysis is a common NLP task in which the objective is to classify text as positive or negative. In this assignment, movie reviews from the IMDB dataset were used to compare conventional vectorization methods with a dense embedding-based approach. The goal was to study the trade-off between accuracy and computation time, and to understand which vectorization technique is most suitable for this dataset.

## 2. Literature Review
### 2.1 What is Word Vectorization?
Word vectorization is the transformation of text into numerical vectors. These vectors can represent either word frequency information or semantic relationships between words. Without vectorization, textual data cannot be processed by machine learning algorithms such as Logistic Regression, SVM, or neural models.

### 2.2 Why is Word Vectorization Important in NLP?
Word vectorization is important because it:

- converts unstructured text into structured numerical features,
- enables machine learning and deep learning models to process language,
- helps capture useful signals such as term importance or semantic similarity,
- directly affects accuracy, memory usage, and computational cost.

### 2.3 Vectorization Techniques
#### 2.3.1 Bag of Words (BoW)
Bag of Words is one of the simplest text representation techniques. It creates a vocabulary of unique words from the corpus and represents each document by counting the occurrence of those words. The method is simple and fast, but it ignores context, grammar, and semantic similarity between words. It usually produces very sparse high-dimensional vectors.

#### 2.3.2 TF-IDF
TF-IDF stands for Term Frequency-Inverse Document Frequency. It improves upon BoW by assigning higher weights to words that are important in a document but not too common across the whole corpus. This helps reduce the influence of very frequent but less informative words. TF-IDF is widely used in classification and information retrieval tasks because it often provides strong baseline performance with low computational cost.

#### 2.3.3 GloVe
GloVe (Global Vectors for Word Representation) is a dense embedding method that learns word vectors from global co-occurrence statistics. Unlike BoW and TF-IDF, GloVe captures semantic similarity between words in a lower-dimensional continuous space. Words with related meanings tend to be closer in vector space. In this work, pre-trained GloVe embeddings were used, and review-level vectors were created by averaging the embeddings of the words present in each review.

#### 2.3.4 Word2Vec and BERT
Word2Vec learns dense word embeddings by predicting surrounding words from context or vice versa. It captures semantic relationships more effectively than count-based models. BERT is a contextual transformer-based model that generates different embeddings for the same word depending on sentence context. Although these methods are important in literature and were discussed as part of the review, the current repository implementation contains experimental results for BoW, TF-IDF, and GloVe only.

## 3. Dataset Description
The dataset used in this assignment is the **IMDB 50K Movie Reviews Dataset**. It contains 50,000 movie reviews labeled as either positive or negative. The dataset is balanced, which makes it suitable for binary sentiment classification without requiring additional balancing techniques.

Key properties of the dataset:

- Total samples: 50,000
- Classes: Positive and Negative
- Task: Binary text classification
- Domain: Movie review sentiment analysis

The dataset was split into training and testing sets using an 80:20 ratio. Therefore, 40,000 reviews were used for training and 10,000 reviews were used for testing.

## 4. Methodology
### 4.1 Preprocessing
Before vectorization, the reviews were cleaned and normalized. The following preprocessing pipeline was used:

1. Convert all text to lowercase.
2. Remove HTML tags.
3. Remove punctuation and special characters.
4. Tokenize the text into words.
5. Remove stopwords using NLTK stopword lists.
6. Apply lemmatization using `WordNetLemmatizer`.

This preprocessing step reduces noise and ensures that similar word forms are treated consistently.

### 4.2 Experimental Setup
The following settings were used in the experiments:

- `max_features = 5000` for BoW and TF-IDF
- Train-test split: 80:20
- Random state: 42
- Classifiers used:
  - Logistic Regression
  - Linear SVM

### 4.3 Conventional Methods
Two conventional sparse vectorization methods were tested:

- **Bag of Words + Logistic Regression / Linear SVM**
- **TF-IDF + Logistic Regression / Linear SVM**

### 4.4 Embedding-Based Method
One embedding-based method was tested:

- **GloVe + Logistic Regression / Linear SVM**

The GloVe vectors were loaded from the pre-trained `glove.6B.100d.txt` file. Each review was represented using the mean of all valid word embeddings present in the review.

## 5. Experimental Results
### 5.1 Performance Table

| Method | Classifier | Accuracy | Precision | Recall | F1-Score | Vectorization Time (s) |
|---|---|---:|---:|---:|---:|---:|
| Bag of Words | Logistic Regression | 0.8741 | 0.8709 | 0.8807 | 0.8758 | 4.49 |
| Bag of Words | Linear SVM | 0.8645 | 0.8616 | 0.8710 | 0.8663 | 4.49 |
| TF-IDF | Logistic Regression | 0.8865 | 0.8779 | 0.9000 | 0.8888 | 3.13 |
| TF-IDF | Linear SVM | 0.8805 | 0.8752 | 0.8897 | 0.8824 | 3.13 |
| GloVe | Logistic Regression | 0.7949 | 0.7989 | 0.7924 | 0.7957 | 8.04 |
| GloVe | Linear SVM | 0.7958 | 0.8000 | 0.7930 | 0.7965 | 8.04 |

### 5.2 Best Performing Model
The best result in this experiment was obtained using **TF-IDF with Logistic Regression**, which achieved:

- Accuracy: **88.65%**
- Precision: **87.79%**
- Recall: **90.00%**
- F1-score: **88.88%**

### 5.3 Observations from Classification Reports
The conventional methods performed strongly on both classes, with balanced precision and recall. TF-IDF improved performance over BoW because it reduced the effect of very frequent and less informative terms. GloVe produced lower scores in this setup because average pooling loses word order and fine-grained contextual sentiment cues. Even though GloVe captures semantic similarity, it was not enough to outperform TF-IDF on this review classification task.

## 6. Comparison and Discussion
### 6.1 Conventional vs Embedding-Based Methods
The results show that the conventional vectorization methods outperformed the embedding-based GloVe approach in this assignment.

- **TF-IDF** provided the highest accuracy and F1-score.
- **BoW** also gave strong results and remained competitive.
- **GloVe** was semantically richer, but its simple averaging strategy reduced its effectiveness for document-level sentiment classification.

### 6.2 Accuracy Comparison
TF-IDF performed better than BoW because it gives higher importance to informative words and lower importance to common words. In sentiment analysis, certain opinion-bearing words such as "excellent", "waste", and "boring" are crucial. TF-IDF highlights such terms more effectively than raw frequency counts.

The GloVe model captured semantic relations, but the approach used here relied on mean pooling over word vectors. This causes loss of sequence information, negation patterns, and sentence-level emphasis. As a result, it underperformed compared to TF-IDF.

### 6.3 Time Complexity Comparison
The vectorization times show that:

- TF-IDF was the fastest at **3.13 seconds**
- BoW required **4.49 seconds**
- GloVe required **8.04 seconds**

This means the best-performing method was also the fastest in the current implementation. GloVe involved loading a large embedding file and generating dense vectors for every review, which increased computation time.

### 6.4 Why TF-IDF Worked Best
TF-IDF worked best for three main reasons:

1. The IMDB dataset is large enough for term-weighting statistics to be highly informative.
2. Sentiment classification depends strongly on discriminative words, which TF-IDF emphasizes.
3. Linear classifiers such as Logistic Regression work extremely well on sparse weighted text vectors.

## 7. Conclusion
This assignment compared three word vectorization approaches for sentiment analysis on the IMDB movie review dataset. Among all tested combinations, **TF-IDF with Logistic Regression** gave the best overall performance, achieving **88.65% accuracy**. Bag of Words also performed well, while GloVe provided lower performance in the current implementation despite offering dense semantic representations.

The experiments demonstrate that advanced embeddings do not automatically guarantee better classification results. For document-level sentiment analysis, carefully tuned conventional methods can still outperform embedding-based approaches when paired with strong linear classifiers. Therefore, for this dataset and setup, TF-IDF is the most practical choice because it offers the best balance of accuracy and computational efficiency.

## 8. Limitations and Future Scope
The current repository demonstrates BoW, TF-IDF, and GloVe-based experiments. Word2Vec and BERT are discussed in the literature review, but their full experimental results are not present in the current codebase. In future work, the study can be extended by:

- implementing Word2Vec using domain-trained embeddings,
- fine-tuning BERT for contextual sentiment analysis,
- testing larger vocabulary sizes and n-grams,
- comparing stemming and lemmatization directly,
- evaluating GPU-based transformer models for improved contextual understanding.

## 9. Team Contribution
The work for this assignment was shared equally between both team members.

- *Vishwak S:* Implemented the preprocessing pipeline, conventional vectorization methods (Bag of Words and TF-IDF), generated results, and organized the project repository.
- *Girisudhan K:* Worked on GloVe-based representation, reviewed the experimental outputs, refined the report content, and finalized the documentation for submission.

Overall contribution:
- *Vishwak S:* 50%
- *Girisudhan K:* 50%

## 10. AI Prompts Used
The following prompts were used only for guidance in structuring explanations and report writing:

1. Explain the difference between Bag of Words, TF-IDF, GloVe, Word2Vec, and BERT in NLP.
2. Help me write a short literature review for an assignment comparing text vectorization techniques.
3. Summarize accuracy, precision, recall, and F1-score for sentiment analysis results.
4. Help convert ML experiment results into a formal academic conclusion.


## 11. Repository Files to Mention in Report
You can cite the following project files in the report:

- Main code: `ml_assignment.py`
- Dataset: `IMDB Dataset.csv`
- Results table: `results/comparison_table.csv`
- Accuracy graph: `results/accuracy_comparison.png`
- Time graph: `results/time_comparison.png`
- Classification reports: `results/classification_reports/`

## 12. Final Submission Checklist
- GitHub repository with code and recent commits
- PDF report generated from this content
- 3 to 4 AI prompts used
- Team contribution section filled with real names
- Result graphs added to the report
- Accuracy table added to the report

