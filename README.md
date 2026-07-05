# 🎬 AI Movie Recommendation System

An intelligent movie recommendation system built using **Python**, **Streamlit**, and **Machine Learning**. The application combines **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Recommendation Model** to provide personalized movie suggestions based on user preferences and rating history.

---

## 📌 Features

- 🎯 Personalized movie recommendations
- 🔍 Search movies by title
- 🎭 Filter movies by genre
- ⭐ Filter recommendations by minimum rating
- 🤖 Hybrid recommendation engine
- 📊 Content-Based similarity search
- 👥 Collaborative Filtering using Item-Based & SVD Matrix Factorization
- 💻 Interactive Streamlit Web Interface
- 💾 Pre-trained model support for faster execution

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Pickle

---

## 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                       # Streamlit Web Application
├── content_based.py             # Content-Based Recommendation
├── collaborative_filtering.py   # Item-Based & SVD Collaborative Filtering
├── hybrid.py                    # Hybrid Recommendation Model
├── train_and_save.py            # Train and Save Models
├── load_model.py                # Load Saved Models
├── demo.py                      # Command Line Demo
├── trained_models.pkl           # Pre-trained Models
├── requirements.txt
├── README.md
│
└── data/
    ├── movies.csv
    ├── ratings.csv
    └── generate_data.py
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git

cd movie-recommendation-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📊 Recommendation Algorithms

### 1. Content-Based Filtering

This model recommends movies by analyzing movie genres.

- Converts genres into TF-IDF vectors
- Creates a user profile from previously rated movies
- Uses Cosine Similarity to find similar movies

**Advantages**

- Works well for new users
- Does not depend on other users' ratings
- Fast recommendations

---

### 2. Collaborative Filtering

Uses user rating history instead of movie metadata.

Implemented using:

- Item-Based Collaborative Filtering
- Matrix Factorization (Truncated SVD)

The system predicts ratings for unseen movies and recommends those with the highest predicted ratings.

---

### 3. Hybrid Recommendation

The hybrid model combines both approaches.

Final Score:

```
Hybrid Score =
0.4 × Content Score +
0.6 × Collaborative Score
```

This improves recommendation quality while reducing the cold-start problem.

---

## 🌟 Application Features

### 🎯 Personalized Recommendations

- Select any user
- Generate top movie recommendations
- View user's previously liked movies

---

### 🔍 Movie Search

Users can

- Search by movie title
- Browse all movies
- View similar movies
- Apply genre filters
- Apply minimum rating filters

---

### 📌 Recommendation Filters

- Genre Selection
- Minimum Average Rating
- Number of Recommendations

---

## 💾 Pre-trained Model

The project includes

```
trained_models.pkl
```

which stores trained recommendation models, allowing instant recommendations without retraining.

Load using

```bash
python load_model.py --user 7 --method hybrid --n 10
```

Available methods

- content
- collaborative
- hybrid

---

## 🏋 Training the Models

To retrain the recommendation models

```bash
python train_and_save.py
```

This trains

- Content-Based Model
- Collaborative Filtering Model
- Hybrid Model

and saves them as

```
trained_models.pkl
```

---

## 🎬 Demo

Run the command-line demonstration

```bash
python demo.py
```

For a specific user

```bash
python demo.py --user 10 --n 5
```

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
scipy
streamlit
```

Install using

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

- Movie poster integration
- IMDb API support
- User authentication
- Deep Learning recommendation models
- Recommendation explanations
- Movie trailers
- User favorites and watchlist
- Deployment on Streamlit Cloud or Render

---

## 👨‍💻 Author

**Nithish Kumar Sakthivel**

Developed as a Machine Learning project demonstrating multiple recommendation techniques and an interactive web application.

---

## 📜 License

This project is intended for educational and learning purposes.
