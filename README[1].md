# Movie Recommendation System Using User Watch History

A B.Tech CSE (Data Science) academic project that recommends personalized movies based on an individual user's previous watch history using content-based genre profiling and transparent scoring.

---

## 1. Project Title
**Movie Recommendation System Using User Watch History**

---

## 2. Introduction
In today's digital era, streaming platforms offer thousands of films. Users often face decision fatigue trying to choose what to watch next. This project implements an intuitive, explainable, and lightweight movie recommendation system written in Python and deployed with a clean Streamlit web user interface.

Instead of opaque black-box machine learning algorithms or heavy distributed computing frameworks (such as Spark or neural networks), this project uses a transparent **Content-Based Filtering** technique based on genre matching and normalized ratings. Every recommendation can be directly traced back to movies the user has already watched.

---

## 3. Problem Statement
Modern recommendation systems frequently rely on complex deep learning or collaborative filtering matrix factorizations (such as ALS or SVD), which have notable drawbacks for small datasets:
- High computational overhead and memory usage.
- The "Cold Start" problem for items without rating matrices.
- "Black-box" opacity: students and evaluators cannot explain *why* a particular movie was recommended.

**The goal of this project is to build a reliable, beginner-friendly, and fully explainable movie recommendation engine** that accurately identifies user genre preferences from their watch records and presents the Top 5 most suitable unwatched movies.

---

## 4. Objectives
- Automatically load and clean movie catalogs and user watch histories from CSV files.
- Profile any selected user by inspecting the genres of their previously watched titles.
- Compute a transparent similarity score for all unwatched movies based on genre overlap and movie rating.
- Filter out already-watched titles to prevent redundant recommendations.
- Present the Top 5 ranked recommendations with clear explanations for why each movie was suggested.
- Provide interactive exploratory data analysis (EDA) and visualizations using Matplotlib.
- Deliver clean code that is easy for a college student to explain during an academic viva presentation.

---

## 5. Proposed System
The proposed system functions entirely on lightweight Python libraries (`pandas`, `streamlit`, and `matplotlib`). It does not require any external database, cloud service, or third-party API.

When a user selects their `User_ID`:
1. The system retrieves all films associated with that user from `watch_history.csv`.
2. It breaks down the genres of those films and counts their frequencies to construct a **User Preference Profile**.
3. It examines all candidate movies from `movies.csv` that the user has **NOT** yet watched.
4. For each candidate movie, it computes:
   - **Genre Match Score:** The overlap between candidate genres and the user's top preferences.
   - **Rating Contribution:** A normalized score based on the movie's IMDB-style rating.
5. It ranks candidates in descending order of total score and outputs the Top 5.

---

## 6. How the Recommendation System Works

### What is Content-Based Recommendation?
Content-based recommendation recommends items to a user by matching user tastes with the features (attributes) of the items. In this project, the primary attributes are movie **Genres** (e.g., Sci-Fi, Action, Drama) and **Ratings**.

### What is Movie Similarity?
Movie similarity measures how close a candidate movie's attributes are to the user's established taste profile. If a user primarily watches Sci-Fi and Adventure movies, a candidate movie containing both Sci-Fi and Adventure is considered highly similar.

### How User Watch History is Used
The user's watch history acts as the training ground for their taste profile. By grouping the genres of all movies the user has watched, we determine which genres appear most frequently (e.g., Sci-Fi: 3, Adventure: 2, Drama: 2).

### How Genres are Compared
For every candidate movie, we inspect its individual genres against the user's preferred genres:
- Each match with a user's preferred genre adds to the similarity score.
- We also weight the match by how strongly the user favors that genre (its frequency in their history).

### How Already-Watched Movies are Removed
Before calculating or ranking candidates, the set of `Movie_ID`s present in the user's watch history is subtracted from the candidate pool:
$$\text{Candidates} = \text{All Movies} \setminus \text{Watched Movies}$$
This strictly guarantees that no movie the user has already seen will ever be recommended.

### How the Top 5 Movies are Selected
After computing the total score for all valid candidates, the candidate list is sorted in descending order by `Similarity Score`, and the first 5 records are displayed.

---

## 7. Recommendation Algorithm

The algorithm is intentionally mathematical, transparent, and easy to explain in a viva.

### Step-by-Step Procedure:
1. **Fetch User History**: Let $W_u$ be the set of movies watched by user $u$.
2. **Compute Genre Weights**:
   For each genre $g$, count its occurrences across all movies in $W_u$:
   $$\text{Freq}(g) = \sum_{m \in W_u} \mathbb{I}(g \in \text{Genres}(m))$$
   Normalize the weight so the user's top genre has weight 1.0:
   $$\text{Weight}(g) = \frac{\text{Freq}(g)}{\max_k(\text{Freq}(k))}$$
3. **Filter Candidates**:
   $$C_u = \{ m \in \text{Movies} \mid m \notin W_u \}$$
4. **Compute Candidate Score**:
   For candidate movie $c \in C_u$:
   $$\text{Genre Score}(c) = \sum_{g \in \text{Genres}(c)} \text{Weight}(g)$$
   $$\text{Rating Contribution}(c) = \frac{\text{Rating}(c)}{10} \times \alpha$$
   *(Default parameter $\alpha = 0.5$ balances genre relevance with overall film quality).*

   $$\text{Total Score}(c) = \text{Genre Score}(c) + \text{Rating Contribution}(c)$$
5. **Rank and Select**:
   Sort $C_u$ descending by $\text{Total Score}$ and take the top 5 movies.

---

## 8. Dataset Description

The project is packaged with two self-contained CSV datasets:

### 1. `movies.csv`
Contains 45 realistic, popular movie titles across diverse genres:
- **`Movie_ID`**: Unique integer identifier (e.g., 1, 2, 3).
- **`Movie_Title`**: Name of the movie (e.g., "Inception", "Arrival").
- **`Genre`**: Pipe-delimited list of genres (e.g., `Sci-Fi|Action|Adventure`).
- **`Rating`**: Rating out of 10.0 (e.g., 8.8, 7.9).
- **`Year`**: Release year of the movie (e.g., 2010, 2016).

### 2. `watch_history.csv`
Contains watch logs across 15 different users:
- **`User_ID`**: Unique integer identifier for the user (e.g., 1, 2, ... 15).
- **`Movie_ID`**: References the `Movie_ID` of the film watched.

---

## 9. Technologies Used
- **Python 3.10+**: Core programming language.
- **Pandas**: Data loading, cleaning, filtering, and aggregation.
- **Streamlit**: Web-based user interface and interactive components.
- **Matplotlib**: Statistical plotting for data distributions and genre analytics.

*Note: No heavy ML frameworks (TensorFlow, PySpark, PyTorch) or external databases are required.*

---

## 10. System Workflow
```
[User Selects User_ID from Dropdown]
                │
                ▼
[Load & Clean CSV Datasets with Pandas]
                │
                ▼
[Extract Movies Watched by User_ID]
                │
                ▼
[Calculate Genre Preference Profile & Weights]
                │
                ▼
[Exclude Watched Movies from Movie Pool]
                │
                ▼
[Score Unwatched Movies: Genre Match + Rating Contribution]
                │
                ▼
[Sort by Similarity Score Descending]
                │
                ▼
[Display Top 5 Recommended Movies with Explanation Reasons]
                │
                ▼
[Render Dataset Summary Metrics & Visualizations]
```

---

## 11. Project Structure
```
movie-recommendation-system/
├── app.py              # Complete Streamlit web application and recommendation logic
├── movies.csv          # Catalog of movies with genres, ratings, and release years
├── watch_history.csv   # User watch records mapping User_ID to Movie_ID
├── requirements.txt    # Minimal dependencies (streamlit, pandas, matplotlib)
└── README.md           # Detailed project documentation and viva preparation guide
```

---

## 12. Installation Instructions

1. **Clone or Download the Project**:
   Ensure all files (`app.py`, `movies.csv`, `watch_history.csv`, `requirements.txt`) are in the same folder.

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 13. How to Run the Application

Execute the following command in your terminal inside the project directory:

```bash
streamlit run app.py
```

Streamlit will launch local and network URLs (typically `http://localhost:8501`). Open the URL in any web browser to use the application.

---

## 14. Example Output

### Case: User 1
- **Watched Movies**:
  - Inception (Sci-Fi, Action, Adventure)
  - Interstellar (Sci-Fi, Drama, Adventure)
  - The Martian (Sci-Fi, Adventure, Drama)
- **Top Preferred Genres**:
  - Sci-Fi (3 titles), Adventure (3 titles), Drama (2 titles)
- **Top 5 Generated Recommendations**:
  1. **Dune** (2021) | Genre: Sci-Fi, Adventure, Action | Rating: 8.0/10 | Reason: Matches your top genres Sci-Fi and Adventure.
  2. **Blade Runner 2049** (2017) | Genre: Sci-Fi, Drama, Action | Rating: 8.0/10 | Reason: Matches your top genres Sci-Fi and Drama.
  3. **Arrival** (2016) | Genre: Sci-Fi, Drama, Mystery | Rating: 7.9/10 | Reason: Matches your top genres Sci-Fi and Drama.
  4. **Gravity** (2013) | Genre: Sci-Fi, Thriller, Drama | Rating: 7.7/10 | Reason: Matches your top genres Sci-Fi and Drama.
  5. **Ex Machina** (2014) | Genre: Sci-Fi, Drama, Thriller | Rating: 7.7/10 | Reason: Matches your top genres Sci-Fi and Drama.

---

## 15. Advantages
1. **Explainable Recommendations**: Every single recommendation is accompanied by a transparent reason based on genre overlap.
2. **No Cold-Start for New Movies**: Any new movie added to `movies.csv` can be recommended immediately as long as its genre and rating are provided.
3. **Completely Self-Contained**: Runs without internet access, third-party API keys, or database configuration.
4. **Lightweight & Fast**: Executes instantaneously on any standard laptop.

---

## 16. Limitations
1. **Vocabulary Limitation**: Content similarity depends on predefined genre tags; films with nuanced styles within the same genre cannot be differentiated without synopsis text.
2. **Serendipity**: Users may experience a "filter bubble" where they only receive movies within genres they have previously watched.

---

## 17. Future Scope
- Adding TF-IDF or Word2Vec on movie overview/plot descriptions.
- Incorporating director and cast metadata into the similarity calculation.
- Allowing users to add new movies or log watch history directly through the UI.
- Deploying to Streamlit Community Cloud for public access.

---

## 18. Conclusion
This project successfully fulfills the requirements of a B.Tech CSE Data Science practical system. It demonstrates practical data manipulation using Pandas, intuitive UI design via Streamlit, statistical charting with Matplotlib, and a robust, fully defendable content-based recommendation algorithm based directly on user watch history.
