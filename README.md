# 🎬 Movie Endpoints & Recommender

An end‑to‑end mini‑project that exposes **movie endpoints** (API-style interactions) and a **content‑based recommender** wrapped in a sleek **Streamlit** UI.  
This README explains the idea, tools, how to run it, and how to extend it.

---

## 🧰 Tech Stack

**Language**
- Python

**Core Libraries**
- Pandas — data loading/cleaning
- NumPy — numeric utilities
- scikit‑learn — TF‑IDF vectorization & cosine similarity
- Streamlit — interactive UI
- Matplotlib — quick visualizations

**Data**
- Public/curated **movie datasets** (e.g., CSV/Parquet with columns such as `movie_id`, `title`, `overview`, `genres`, `year`, `rating`)

**Platform**
- **Streamlit** (local or Streamlit Community Cloud for quick deploys)

---

## 🗺️ What you’ll find in the app

The app is divided into **three main charts/sections**:

1) **Dataset Filter Overview**  
   Understand the **filter(s)** currently applied to the dataset (e.g., by year, minimum rating, language, or availability of overview/genres).  
   - Shows counts and distributions to confirm the working subset.  
   - Helps explain *why* certain titles appear (or don’t) in the recommendations.

2) **Movie Recommender**  
   A **content‑based** engine that uses **TF‑IDF** on textual fields (title/overview/genres/keywords) and **cosine similarity** to surface movies similar to a chosen seed (by title or id).  
   - Returns top‑K similar movies with optional scores.  
   - Sensible fallbacks for empty/rare inputs.

3) **Interactive Endpoints**  
   A hands‑on panel that mimics **API endpoints** to fetch a movie, search by query, and request recommendations.  
   - Great for testing and demoing “endpoint‑like” behaviors without leaving Streamlit.  
   - Each action displays response payloads (e.g., JSON‑like dicts) so you can prototype client integration.

> Tip: If you have an actual FastAPI service, these controls can be pointed to your base URL instead of local UI functions.

---

## 🧠 How the recommender works (high‑level)

- **Featurization:** Build a text string per movie (e.g., `title + overview + genres`).  
- **Vectorization:** Use `TfidfVectorizer` (optionally with stopwords removal, min_df, max_features).  
- **Similarity:** Compute **cosine similarity** between the seed movie’s vector and the rest.  
- **Ranking:** Return the top‑K similar movies as recommendations.  
- **Possible tweaks:** weighting fields (e.g., overview 0.6, genres 0.3, keywords 0.1), tie‑breakers by rating/year, and diversity re‑ranking.

---

## 📦 Project Structure (suggested)

```
.
├─ data/                      # datasets (CSV/Parquet)
├─ models/                    # saved artifacts (optional: tfidf.pkl, matrix.npz)
├─ app/
│  ├─ recommender.py          # TF-IDF + similarity logic
│  ├─ data_utils.py           # loading, cleaning, filters
│  ├─ charts.py               # plotting helpers (matplotlib)
│  └─ config.py               # constants / toggles
├─ streamlit_app.py           # Streamlit UI (3 main sections)
├─ requirements.txt
└─ README.md
```

---

## 🔧 Setup & Installation

> **Prerequisites:** Python 
```bash
# 1) Create & activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
streamlit
pandas
numpy
scikit-learn
matplotlib
python-dotenv
```

---

## ▶️ Run the App (Streamlit)

```bash
streamlit run streamlit_app.py
```

- The app opens in your browser ( `http://localhost:8501`).  
- Use the left sidebar to select **filters**, choose a **seed movie**, and test the **interactive endpoints**.

---

## 🔌 “Endpoint‑like” Actions (inside Streamlit)

These actions simulate simple REST behaviors directly in the UI (or can proxy to a real API if configured):

- **GET /movies/{movie_id}** → returns details for a specific movie id  
- **GET /search?query=...** → searches movies by text query  


> To connect a real API, add a base URL in your config or environment, and replace the local functions with `requests` calls.

---

## 📊 The 3 Charts (example behavior)

1. **Filter Summary Chart** (Matplotlib)  
   - Bar/line charts for count by year, rating distribution, or genre coverage.  
   - A badge showing resulting dataset size after filters.

2. **Recommender Results**  
   - Table or cards of top‑K recommendations with similarity score (optional).  
   - Optional small histogram of similarity scores for transparency.

3. **Endpoint Responses**  
   - JSON‑like boxes displaying the result of “get/search/recommend” calls.  
   - Copy‑to‑clipboard buttons for quick testing (optional).

---
## Sections
<img width="1539" height="642" alt="image" src="https://github.com/user-attachments/assets/5a2131a1-1662-47cd-a3d6-5cc812cab96a" />
<img width="903" height="433" alt="image" src="https://github.com/user-attachments/assets/4abcf28f-4959-4b3a-b3ef-77682430b4c4" />
<img width="1755" height="559" alt="image" src="https://github.com/user-attachments/assets/0222bf33-9a9b-49c7-b8a2-4d3431da8c96" />


## 🧪 Quality checks & tips

- **Null & duplicates:** Drop/clean before vectorization.  
- **Stopwords & normalization:** Lowercase, strip accents, remove stopwords for cleaner features.  
- **Cold start:** When seed not found, fall back to popular titles or a nearest text match.  
- **Performance:** Cache vectorizer & matrix, or persist artifacts under `models/`.

---

## 🚢 Deploying on Streamlit Cloud

1. Push your repo (include `requirements.txt`).  
2. Add your `data/` (or remote path & load from URL/storage).  
3. In Streamlit Cloud, point to `streamlit_app.py` and set any environment variables.  
4. (Optional) Store large artifacts in cloud storage and download at startup.
