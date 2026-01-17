# Hybrid Movie Recommendation System

### Neural Aspect-Based Filtering + Collaborative Filtering (SVD)

This project implements a **Hybrid Recommendation Engine** that combines the strengths of content-based "aspect" analysis with traditional Collaborative Filtering. By using a Neural Network, the system learns complex relationships between movie metadata (genres, keywords, overviews) and user preferences, which is then blended with Matrix Factorization (SVD) to provide highly personalized movie suggestions.

## 🚀 Overview

Most recommendation systems suffer from either the "Cold Start" problem (Collaborative Filtering) or a lack of serendipity (Content-Based). This system bridges that gap by:

1. **Extracting "Aspects":** Using NLP to identify word-pair features from movie descriptions.
2. **Learning Preferences:** Training a PyTorch-based neural network to predict how much a user will like a specific movie "aspect."
3. **Latent Factor Modeling:** Using Singular Value Decomposition (SVD) to find hidden patterns in user-item interactions.
4. **Weighted Hybridization:** Combining both scores to produce a final ranked list.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Machine Learning:** PyTorch (Neural Networks), Scikit-Learn (SVD, TF-IDF, Cosine Similarity)
* **Data Processing:** Pandas, NumPy
* **NLP:** Regular Expressions (Regex) for feature extraction

---

## 📂 Project Structure

* `finaldataset.csv`: The core dataset containing `userId`, `movieId`, `rating`, `timestamp`, and metadata (`genres`, `keywords`, `overview`).
* `finalmodel.py`: The main Python script containing the preprocessing, training, and evaluation logic.
* `tiny_aspect_nn_weights.pth`: Saved weights for the Neural Network, allowing for quick inference without retraining.

---

## ⚙️ How It Works

### 1. Preprocessing & Aspect Extraction

The system cleans the textual metadata and uses a regex-based approach to extract meaningful bigrams (word pairs). These "aspects" are then converted into a numerical format using a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency), limited to the top 1,000 features to maintain performance.

### 2. The Tiny Neural Network (TinyNN)

Instead of simple linear similarity, we use a Feed-Forward Neural Network with:

* **Input Layer:** 1,000 dimensions (TF-IDF features).
* **Hidden Layer:** 32 neurons with ReLU activation.
* **Output Layer:** A single scalar predicting the user's rating.
This allows the model to capture non-linear interests in specific movie themes or genres.

### 3. SVD Collaborative Filtering

We create a user-item matrix and decompose it into 50 latent factors. This identifies "users like you" and recommends movies based on communal behavior.

### 4. Hybrid Scoring Formula

The final recommendation score is calculated as:



*Default weights: 60% Aspect-based, 40% Collaborative Filtering.*

---

## 📊 Performance Metrics

The model is evaluated using standard Information Retrieval metrics:

* **Precision@5:** How many of the top 5 recommendations were actually relevant.
* **Recall@5:** How many of the user's actual liked movies were captured in the top 5.
* **NDCG@5:** (Normalized Discounted Cumulative Gain) Measures the quality of the ranking (placing highly relevant movies at the very top).

---

## 🏃 Instructions to Run

1. **Install Dependencies:**
```bash
pip install pandas numpy torch scikit-learn

```
2. **Dataset Link:** 
```bash
https://drive.google.com/file/d/1vmlickoS6YJQiVeaBSAv0xMBJ4eF6kWa/view?usp=sharing
```
3. **Prepare Data:** Ensure `finaldataset.csv` is in the root directory.
4. **Execute:**
```bash
python recommendation_engine.py

```


5. **Get Recommendations:** When prompted, enter a `User ID` to see a personalized list of 5 movies and their calculated hybrid scores.
