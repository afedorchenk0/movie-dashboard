import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Industry Dashboard", layout="wide")
st.title("Дашборд: фільми 1980-2020")

@st.cache_data
def load_data():
    return pd.read_csv("movies_enriched.csv")

df = load_data()

st.write(
    "Аналіз датасету Movie Industry: 7668 фільмів із даними про бюджет, "
    "збори, рейтинг, жанр та інше."
)

with st.expander("Переглянути весь датасет"):
    st.dataframe(df)

st.sidebar.header("Фільтри")

year_min = int(df["year"].min())
year_max = int(df["year"].max())
years = st.sidebar.slider("Роки виходу фільму", year_min, year_max, (year_min, year_max))

all_genres = sorted(df["genre"].dropna().unique())
genres = st.sidebar.multiselect("Жанри", all_genres, default=all_genres)

fdf = df[df["year"].between(years[0], years[1]) & df["genre"].isin(genres)]

if len(fdf) == 0:
    st.warning("За обраними фільтрами фільмів не знайдено.")
    st.stop()

st.write(f"Відібрано фільмів: **{len(fdf)}** із {len(df)}")

# Графік 1
st.subheader("1. Окупність фільмів за десятиліттями")

roi_by_decade = fdf.groupby("decade")["roi"].median().dropna()

if roi_by_decade.empty:
    st.info("Для обраних фільтрів немає фільмів з відомими бюджетом і зборами.")
else:
    fig1, ax1 = plt.subplots()
    roi_by_decade.plot(kind="bar", ax=ax1)
    ax1.set_title("Медіанна окупність фільмів за десятиліттями")
    ax1.set_xlabel("Десятиліття")
    ax1.set_ylabel("Окупність (збори / бюджет)")
    ax1.tick_params(axis="x", rotation=0)
    st.pyplot(fig1)
    plt.close(fig1)

st.caption(
    "Окупність показує, у скільки разів збори перевищили бюджет. Рахується медіана, "
    "бо кілька блокбастерів сильно завищують середнє. Фільми без бюджету чи зборів не враховано. "
    "Десятиліття 2020 неповне: у ньому лише 25 фільмів."
)

# Графік 2
st.subheader("2. Чи впливає довжина фільму на рейтинг")

influence_of_length = fdf.groupby("length_type", observed=True)["score"].agg(["mean", "count"])

if influence_of_length["count"].sum() == 0:
    st.info("Для обраних фільтрів немає фільмів з відомими тривалістю і рейтингом.")
else:
    fig2, ax2 = plt.subplots()
    influence_of_length["mean"].plot(kind="bar", ax=ax2, color="coral")
    ax2.set_title("Середній рейтинг за тривалістю фільму")
    ax2.set_xlabel("Тривалість фільму")
    ax2.set_ylabel("Середній рейтинг")
    ax2.tick_params(axis="x", rotation=0)
    st.pyplot(fig2)
    plt.close(fig2)

st.caption(
    "Довші фільми мають вищий середній рейтинг. Короткі фільми (до 90 хв) мають середній рейтинг 5.9, "
    "середні (91-120 хв) 6,32, довгі (понад 120 хв) 7,04. Групи великі (від 1114 до 5128 фільмів), "
    "але це лише зв'язок, а не доказ того, що тривалість впливає на оцінку."
)

# Графік 3
st.subheader("3. Чи пов'язані бюджет і збори?")

budget_gross_df = fdf.dropna(subset=["budget", "gross"])

if budget_gross_df.empty:
    st.info("Для обраних фільтрів немає фільмів з відомими бюджетом і зборами.")
else:
    fig3, ax3 = plt.subplots()
    budget_gross_df.plot(kind="scatter", x="budget", y="gross", alpha=0.3, color="purple", ax=ax3)
    ax3.set_title("Залежність зборів від бюджету")
    ax3.set_xlabel("Бюджет")
    ax3.set_ylabel("Збори")
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Бюджет і збори помітно пов'язані. Коефіцієнт кореляції дорівнює 0,74, тобто в цілому дорожчі фільми збирають більше. "
        "Проте розкид великий: серед фільмів із найбільшими бюджетами збори дуже різні, тож високий бюджет не гарантує успіху. "
        "Розрахунок зроблено лише для фільмів, у яких вказані і бюджет, і збори."
    )
