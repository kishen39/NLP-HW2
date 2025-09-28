# NLP-HW2
---
**Kishen
700762472**
---

# 📊 Q3 Overview
The system evaluates **classification performance** using a confusion matrix from an animal classification task where 90 animals were classified into Cat, Dog, or Rabbit. The tool computes comprehensive evaluation metrics to assess model effectiveness across different animal categories.

1️⃣ **Input Data:**
```python
System \ Gold  Cat  Dog  Rabbit
Cat            5    10    5
Dog            15   20    10
Rabbit         0    15    10
```
🧮 **Implementation Features**
Core Functionality:
Per-class Metrics: Precision and recall calculations for each category
Macro-averaging: Equal weight averaging across all classes
Micro-averaging: Pooled metrics considering class distribution
Support Statistics: TP, FP, FN, TN computations

2️⃣Key Computations:
```python
# Precision: TP / (TP + FP)
# Recall: TP / (TP + FN)
# Macro-average: Mean of per-class metrics
# Micro-average: Global pooled metrics
```

📈 Sample Results
```python
Per-class Performance:
Cat: Precision=0.2500, Recall=0.2500

Dog: Precision=0.4444, Recall=0.4444

Rabbit: Precision=0.4000, Recall=0.4000

Aggregate Metrics:
Macro-average Precision: 0.3648

Macro-average Recall: 0.3648

Micro-average Precision: 0.3889

Micro-average Recall: 0.3889

Overall Accuracy: 38.89%
```
🔍 **Performance Insights**
The evaluation reveals:

Cat classification shows the lowest performance with significant misclassifications

Dog and Rabbit show moderate but similar performance levels

Overall model accuracy of 38.89% indicates substantial room for improvement

Consistent precision-recall values suggest balanced performance across metrics

# 📊 **Q8 Overview**
This project implements a **Bigram Language Model** based on the "I love NLP corpus" activity, computing probabilities and evaluating sentence likelihoods.

1️⃣ **Read Training Corpus**
```python
<s> I love NLP </s>
<s> I love deep learning </s>
<s> deep learning is fun </s>
```
2️⃣ **Compute Unigram and Bigram Counts**
3️⃣ **Estimate Bigram Probabilities using MLE**
4️⃣ **Implement Probability Calculation Function**

```python
def sentence_probability(self, sentence):
    tokens = sentence.split()
    probability = 1.0
    for i in range(len(tokens) - 1):
        bigram_prob = self.calculate_bigram_probability(tokens[i], tokens[i+1])
        probability *= bigram_prob
    return probability
```
5️⃣ **Test Function on Sentences**
Test Sentence 1: 
```python
<s> I love NLP </s>
Overall Probability: 0.333333
```

Test Sentence 2:
```python
<s> I love deep learning </s>
Overall Probability: 0.166667
```

6️⃣ **Model Preference Results**

The model prefers:
```python
<s> I love NLP </s>
```
Reason: Higher probability (0.333333 > 0.166667) 

🧮 **Mathematical Foundation**

Bigram Probability: P(w2|w1) = count(w1, w2) / count(w1)

Sentence Probability: P(sentence) = Π P(w_i|w_{i-1})