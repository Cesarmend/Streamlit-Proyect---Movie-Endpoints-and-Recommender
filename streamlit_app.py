# streamlit_app.py
import ast
import difflib
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="ML Movie Recommender 🍿",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= GLOBAL STYLES =========
st.markdown("""
<style>
/* Big centered main title */
.main-title {
  text-align: center;
  font-size: 3rem;
  line-height: 1.2;
  margin: .6rem 0 .2rem 0;
}
/* Bigger, bolder tabs */
.stTabs [data-baseweb="tab"] p {
  font-size: 1.2rem;
  font-weight: 700;
}
/* Centered subheader + caption */
.center-subheader {
  text-align: center;
  font-size: 1.75rem;
  margin: .25rem 0 0 0;
}
.center-caption {
  text-align: center;
  font-size: 1rem;
  margin: .15rem 0 1rem 0;
}
/* Center buttons (Recommend, etc.) */
div.stButton > button {
  display: block;
  margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# ========= PATHS =========
DATA_PATH = Path("Datasets/df_test_output.csv")

# ========= SMALL HELPERS =========
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def primary_lang(x):
    """Return first language from list/string; handles '["English","Spanish"]' strings as well."""
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return str(x[0]).strip() if x else None
    s = str(x).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            s = str(parsed[0])
    except Exception:
        pass
    for sep in [",", "|", ";", "/"]:
        if sep in s:
            s = s.split(sep)[0]
            break
    return s.strip()

def normalize_to_chart_label(s: str):
    if not s:
        return None
    low = strip_accents(s).lower().strip()
    if low in {"english", "ingles", "inglesa"}:
        return "English"
    if low in {"french","frances","francais","francesa","franceses","frances "}:
        return "Francés"
    if low in {"spanish","espanol","espanola","castellano"}:
        return "Español"
    if low in {"italian","italiano","italiana"}:
        return "Italiano"
    if low in {"japanese","japones","japonesa"}:
        return "Japonés"
    if low in {"portuguese","portugues","portuguesa","portugueses"}:
        return "Portugués"
    return s.strip()

def explode_names(series: pd.Series) -> pd.Series:
    """Flatten list-like or delimited strings into a Series of names."""
    all_names = []
    for raw in series.dropna().astype(str):
        items = None
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                items = parsed
        except Exception:
            pass
        if items is None:
            items = re.split(r"[|,;/]", raw)
        all_names.extend([str(x).strip() for x in items if str(x).strip()])
    return pd.Series(all_names, dtype="string")

def sample_top_movies(df: pd.DataFrame, n: int = 7) -> list:
    s = (df.sort_values("popularity", ascending=False)["title"]
           .astype(str).dropna().drop_duplicates())
    return s.head(n).tolist()

def sample_top_actors(df: pd.DataFrame, n: int = 7) -> list:
    names = explode_names(df["actor_names"])
    vc = names.value_counts()
    return vc.head(n).index.tolist()

def sample_top_directors(df: pd.DataFrame, n: int = 7) -> list:
    vc = (df["director_name"].dropna().astype(str).str.strip()
            .replace("", np.nan).dropna().value_counts())
    return vc.head(n).index.tolist()

def examples_caption(examples: list, label: str = "e.g."):
    if examples:
        st.caption(f"{label} " + " · ".join(examples))

# ========= LOAD DATA & MODEL (cached) =========
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).reset_index(drop=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["primary_language"] = (
        df["language_names"].apply(primary_lang).apply(normalize_to_chart_label)
    )
    return df

@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame):
    """Build TF-IDF + cosine similarity matrix for the recommender."""
    key_cols = ["genre_names", "actor_names", "overview", "title"]
    for c in key_cols:
        df[c] = df[c].fillna("")
    text = (
        df["genre_names"].astype(str) + " " +
        df["actor_names"].astype(str) + " " +
        df["title"].astype(str) + " " +
        df["overview"].astype(str)
    )
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    sim = cosine_similarity(X)  # square matrix aligned to df.index
    titles = df["title"].astype(str).tolist()
    return vectorizer, sim, titles

df = load_data(DATA_PATH)
_, SIM, TITLES = build_model(df)

# ========= “ENDPOINT” LOGIC =========
MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}
DAYS_ES = {
    "lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "domingo": 6
}

def ep_cantidad_filmaciones_mes(df: pd.DataFrame, mes: str) -> str:
    m = (mes or "").strip().lower()
    if m not in MONTHS_ES:
        raise ValueError("Mes no válido")
    month_number = MONTHS_ES[m]
    count = df[df["release_date"].dt.month == month_number].shape[0]
    return f"{count} peliculas han sido lanzadas en {m.capitalize()}"

def ep_cantidad_filmaciones_dia(df: pd.DataFrame, dia: str) -> str:
    d = (dia or "").strip().lower()
    if d not in DAYS_ES:
        raise ValueError("Día no válido")
    day_number = DAYS_ES[d]
    count = df[df["release_date"].dt.weekday == day_number].shape[0]
    return f"{count} peliculas han sido lanzadas en {d.capitalize()}"

def ep_puntaje_por_titulo(df: pd.DataFrame, titulo: str) -> str:
    t = (titulo or "").strip().lower()
    row = df[df["title"].str.lower() == t]
    if row.empty:
        raise ValueError("Pelicula no encontrada")
    r = row.iloc[0]
    return (f"La pelicula '{r['title']}' fue estrenada en el año {int(r['release_year'])} "
            f"con un puntaje de popularidad de {round(float(r['popularity']), 2)}")

def ep_votos_titulo(df: pd.DataFrame, titulo: str) -> str:
    t = (titulo or "").strip().lower()
    row = df[df["title"].str.lower() == t]
    if row.empty:
        raise ValueError("Pelicula no encontrada")
    r = row.iloc[0]
    if int(r["vote_count"]) < 2000:
        raise ValueError("La pelicula no cumple con el minimo de 2000 valoraciones")
    return (f"La pelicula '{r['title']}' fue estrenada en el año {int(r['release_year'])} "
            f"con un total de {int(r['vote_count'])} valoraciones y un promedio de {float(r['vote_average'])}")

def ep_reporte_actor(df: pd.DataFrame, nombre_actor: str):
    a = (nombre_actor or "").strip().lower()
    films = df[df["actor_names"].str.lower().str.contains(a, na=False)]
    if films.empty:
        raise ValueError("Actor no encontrado o no tiene películas en el dataset")
    film_count = films.shape[0]
    total_return = float(films["return"].sum())
    avg_return = total_return / film_count if film_count else 0.0
    table = films[["title", "release_date", "budget", "revenue", "return"]].copy()
    return {
        "msg": (f"El actor '{nombre_actor.title()}' ha participado en {film_count} filmaciones, "
                f"el mismo ha conseguido una ganancia de {total_return:,.2f} USD "
                f"con un promedio de {avg_return:,.2f} USD por filmación"),
        "table": table
    }

def ep_reporte_director(df: pd.DataFrame, nombre_director: str):
    d = (nombre_director or "").strip().lower()
    films = df[df["director_name"].str.lower() == d]
    if films.empty:
        raise ValueError("Director no encontrado o no tiene peliculas en el dataset")
    total_return = float(films["return"].sum())
    table = films[["title", "release_date", "budget", "revenue", "return"]].copy()
    return {
        "header": f"Director: {nombre_director.title()} — Total return: {total_return:,.2f}",
        "table": table
    }

# ========= HEADER =========
st.markdown('<h1 class="main-title">🎥 ML Movie Recommender 🍿</h1>', unsafe_allow_html=True)
st.caption("3 tabs to interact with: • Charts(Analysis by language) • Recommender • Endpoint-style actions with clear (e.g.) examples")
# ========= HEADER =========
# ---- label before tabs ----
st.markdown("""
<style>
.tabs-lead {
  font-size: 1.15rem;
  font-weight: 700;
  margin: .25rem 0 .5rem 0;
  text-align: left !important;  /* left align */
}
</style>
<p class="tabs-lead">Select a section between: | Charts | Recommender | Endpoints |</p>
""", unsafe_allow_html=True)

# ---- center the tabs ----
st.markdown("""
<style>
/* Center the tab headers */
.stTabs [role="tablist"] {
  display: flex;
  justify-content: center;   /* center the whole group */
  gap: 0.8rem;               /* space between tabs */
  width: 100%;
}
/* Prevent tabs from stretching */
.stTabs [data-baseweb="tab"] {
  flex: 0 0 auto;
}
</style>
""", unsafe_allow_html=True)

# ========= TABS =========
tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔍Recommender", "🧪 Endpoint Actions"])

# ----- TAB 1: CHARTS -----
with tab1:
    images = [
        {"title": "Language Distribution", "path": "Images/idiomas_distribution.png"},
        {"title": "High Rated Movies Languages (excluding English)", "path": "Images/high_lang_donut.png"},
    ]

    if "chart_idx" not in st.session_state:
        st.session_state.chart_idx = 0

    # Centered Prev / Next buttons (group centered)
    spL, cPrev, cGap, cNext, spR = st.columns([4, 1.2, 0.6, 1.2, 4])
    with cPrev:
        prev_clicked = st.button("◀ Prev", use_container_width=True)
    with cNext:
        next_clicked = st.button("Next ▶", use_container_width=True)

    if prev_clicked:
        st.session_state.chart_idx = (st.session_state.chart_idx - 1) % len(images)
    if next_clicked:
        st.session_state.chart_idx = (st.session_state.chart_idx + 1) % len(images)

    current = images[st.session_state.chart_idx]
    st.markdown(
        f"<h2 style='margin:.25rem 0 .75rem 0;text-align:center'>{current['title']}</h2>",
        unsafe_allow_html=True
    )
    img_path = Path(current["path"])

    if img_path.exists():
        # Option B: fit-to-column by narrowing the middle column
        # Tweak the ratios to change image size: larger side columns => smaller image
        l, m, r = st.columns([5, 2, 5])  # try [6,2,6] or [4,2,4] to adjust
        with m:
            st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"Image not found at {img_path}. Place it under /Images or update the path.")


# ----- TAB 2: RECOMMENDATIONS -----
with tab2:
    # Headings centered
    st.markdown('<div class="center-subheader">📃 Get 5 suggested movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-caption">based on your entry</div>', unsafe_allow_html=True)

    # Centered input + button
    c1, c2, c3 = st.columns([1, 2, 1])
    run_clicked = False
    with c2:
        st.text_input("Enter a movie title", placeholder="e.g., The Matrix", key="movie_query")
        examples_caption(sample_top_movies(df, 7), "e.g.")
        run_clicked = st.button("✅ Recommend ▶")

    if run_clicked:
        query = st.session_state.get("movie_query", "")
        if not query.strip():
            st.warning("Please type a movie title.")
        else:
            candidates = difflib.get_close_matches(query, TITLES, n=1, cutoff=0.4)
            if not candidates:
                st.error("Title not found. Try another.")
            else:
                ref = candidates[0]
                idx = df[df["title"] == ref].index[0]
                scores = list(enumerate(SIM[idx]))
                ranked = sorted(scores, key=lambda x: x[1], reverse=True)

                recs = []
                for j, _ in ranked:
                    t = TITLES[j]
                    if t.lower() != query.lower():
                        recs.append(t)
                    if len(recs) >= 5:
                        break

                st.success(f"Top recommendations for **{ref}**")
                for k, r in enumerate(recs, 1):
                    st.write(f"**{k}.** {r}")

# ----- TAB 3: ENDPOINT ACTIONS (LOCAL) -----
with tab3:
    st.markdown("""
**What these do**
- **Month films** — total releases for a Spanish month name (`enero`…`diciembre`).
- **Day films** — total releases for a Spanish weekday (`lunes`…`domingo`).
- **Score by title** — popularity score and release year of a movie.
- **Votes by title** — year, vote count (requires ≥ 2000), and vote average.
- **Actor report** — total films, total/avg return + their films.
- **Director report** — exact director match + films and total return.
    """)

    c1, c2 = st.columns(2)

    # Pre-computed dataset-based examples (deterministic)
    movie_examples = sample_top_movies(df, 7)
    actor_examples = sample_top_actors(df, 7)
    director_examples = sample_top_directors(df, 7)

    # 1) Month
    with c1:
        with st.form("form_mes"):
            mes = st.text_input("📆 Mes (español):", placeholder="e.g., enero, febrero, marzo", key="mes_input")
            st.caption("e.g. enero · febrero · marzo · abril · mayo · junio · julio")
            run_mes = st.form_submit_button("▶ Run month")
        if run_mes:
            try:
                st.success(ep_cantidad_filmaciones_mes(df, mes))
            except Exception as e:
                st.error(str(e))

    # 2) Day
    with c1:
        with st.form("form_dia"):
            dia = st.text_input("📆 Día (español):", placeholder="e.g., lunes, martes, miércoles", key="dia_input")
            st.caption("e.g. lunes · martes · miercoles · jueves · viernes · sabado · domingo")
            run_dia = st.form_submit_button("▶ Run day")
        if run_dia:
            try:
                st.success(ep_cantidad_filmaciones_dia(df, dia))
            except Exception as e:
                st.error(str(e))

    # 3) Score by title
    with c1:
        with st.form("form_score"):
            titulo_score = st.text_input("📼 Título (puntaje/popularidad):", key="score_input",
                                         placeholder="e.g., Titanic")
            examples_caption(movie_examples, "e.g.")
            run_score = st.form_submit_button("▶ Run score")
        if run_score:
            try:
                st.info(ep_puntaje_por_titulo(df, titulo_score))
            except Exception as e:
                st.error(str(e))

    # 4) Votes by title
    with c2:
        with st.form("form_votes"):
            titulo_votes = st.text_input("🔍 Título (≥ 2000 votos):", key="votes_input",
                                         placeholder="e.g., Avatar")
            examples_caption(movie_examples, "e.g.")
            run_votes = st.form_submit_button("▶ Run votes")
        if run_votes:
            try:
                st.info(ep_votos_titulo(df, titulo_votes))
            except Exception as e:
                st.error(str(e))

    # 5) Actor report
    with c2:
        with st.form("form_actor"):
            actor = st.text_input("🎭 Nombre del actor:", key="actor_input",
                                  placeholder="e.g., Tom Hanks")
            examples_caption(actor_examples, "e.g.")
            run_actor = st.form_submit_button("▶ Run actor report")
        if run_actor:
            try:
                res = ep_reporte_actor(df, st.session_state.get("actor_input", actor))
                st.success(res["msg"])
                st.dataframe(res["table"], use_container_width=True)
            except Exception as e:
                st.error(str(e))

    # 6) Director report
    with c2:
        with st.form("form_director"):
            director = st.text_input("🎬 Nombre del director:", key="director_input",
                                     placeholder="e.g., Steven Spielberg")
            examples_caption(director_examples, "e.g.")
            run_director = st.form_submit_button("▶ Run director report")
        if run_director:
            try:
                director_query = st.session_state.get("director_input") or director
                res = ep_reporte_director(df, director_query)
                st.success(res["header"])
                st.dataframe(res["table"], use_container_width=True)
            except Exception as e:
                st.error(str(e))

# ========= FOOTER =========
st.divider()
st.markdown('<div class="center-caption">® Demo of Machine Learning sentiment analysis recommender.</div>',
            unsafe_allow_html=True)
