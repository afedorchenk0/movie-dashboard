import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Industry Dashboard", layout="wide")
st.title("Дашборд: фільми 1980-2020")


def fmt(x, digits=2):
    #Число з комою замість крапки: 6.32 -> 6,32 (для порожнього значення: н/д)
    if pd.isna(x):
        return "н/д"
    return f"{x:.{digits}f}".replace(".", ",")


@st.cache_data
def load_data():
    data = pd.read_csv("movies_enriched.csv")
    # Після збереження в CSV категорії губляться, тому задаємо порядок груп вручну
    order = ["короткий", "середній", "довгий"]
    data["length_type"] = pd.Categorical(data["length_type"], categories=order, ordered=True)
    return data


df = load_data()

st.write(
    f"Аналіз датасету Movie Industry: {len(df)} фільмів із даними про бюджет, "
    "збори, рейтинг, жанр та інше. Оригінальний датасет: "
    "[Kaggle](https://www.kaggle.com/datasets/danielgrijalvas/movies)."
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
        "У десятилітті 2020 у датасеті лише 25 фільмів, тож воно неповне."
    )

# Графік 2
st.subheader("2. Чи впливає тривалість фільму на рейтинг?")

influence_of_length = fdf.groupby("length_type", observed=True)["score"].agg(["mean", "count"])

if influence_of_length["count"].sum() == 0:
    st.info("Для обраних фільтрів немає фільмів з відомими тривалістю і рейтингом.")
else:
    fig2, ax2 = plt.subplots()
    influence_of_length["mean"].plot(kind="bar", ax=ax2, color="coral")
    ax2.set_title("Середній рейтинг IMDb за тривалістю фільму")
    ax2.set_xlabel("Тривалість: короткий (до 90 хв), середній (91-120 хв), довгий (понад 120 хв)")
    ax2.set_ylabel("Середній рейтинг (0-10)")
    ax2.tick_params(axis="x", rotation=0)
    st.pyplot(fig2)
    plt.close(fig2)

    parts = [
        f"{name}: {fmt(row['mean'])} (фільмів: {int(row['count'])})"
        for name, row in influence_of_length.iterrows()
    ]
    st.caption(
        "Середній рейтинг за групами для обраних фільмів: " + "; ".join(parts) + ". "
        "Це лише зв'язок, а не доказ того, що тривалість впливає на оцінку."
    )

# Графік 3
st.subheader("3. Чи пов'язані бюджет і збори?")

budget_gross_df = fdf.dropna(subset=["budget", "gross"])

if budget_gross_df.empty:
    st.info("Для обраних фільтрів немає фільмів з відомими бюджетом і зборами.")
else:
    plot_df = budget_gross_df.assign(
        budget_m=budget_gross_df["budget"] / 1_000_000,
        gross_m=budget_gross_df["gross"] / 1_000_000,
    )
    fig3, ax3 = plt.subplots()
    plot_df.plot(kind="scatter", x="budget_m", y="gross_m", alpha=0.3, color="purple", ax=ax3)
    ax3.set_title("Залежність зборів від бюджету")
    ax3.set_xlabel("Бюджет, млн $")
    ax3.set_ylabel("Збори, млн $")
    st.pyplot(fig3)
    plt.close(fig3)

    corr = budget_gross_df["budget"].corr(budget_gross_df["gross"])
    corr_text = f"Коефіцієнт кореляції для обраних фільмів: {fmt(corr)}. " if pd.notna(corr) else ""
    st.caption(
        corr_text +
        "Чим ближче коефіцієнт до 1, тим сильніше дорожчі фільми збирають більше. "
        "Розкид великий: серед фільмів з найбільшими бюджетами збори дуже різні, "
        "тож високий бюджет не гарантує успіху. "
        "Розрахунок зроблено лише для фільмів, у яких вказані і бюджет, і збори."
    )

# Графік 4
st.subheader("4. Чим особливі переможці «Оскара»?")

n_winners = int(fdf["oscar_winner"].sum())

if n_winners == 0:
    st.info("Серед обраних фільмів немає переможців «Оскара». Змініть фільтри.")
else:
    labels = ["Без «Оскара»", "Переможці «Оскара»"]
    oscar_stats = (
        fdf.groupby("oscar_winner")[["budget", "gross", "score"]]
        .median()
        .reindex([False, True])
    )

    col1, col2 = st.columns(2)

    with col1:
        money = oscar_stats[["budget", "gross"]].T / 1_000_000
        money.columns = labels
        money.index = ["Бюджет", "Збори"]
        fig4a, ax4a = plt.subplots()
        money.plot(kind="bar", ax=ax4a)
        ax4a.set_title("Медіанні бюджет і збори")
        ax4a.set_xlabel("")
        ax4a.set_ylabel("Медіана, млн $")
        ax4a.tick_params(axis="x", rotation=0)
        st.pyplot(fig4a)
        plt.close(fig4a)

    with col2:
        score = oscar_stats["score"]
        score.index = labels
        fig4b, ax4b = plt.subplots()
        score.plot(kind="bar", ax=ax4b, color=["gray", "gold"])
        ax4b.set_title("Медіанний рейтинг IMDb")
        ax4b.set_xlabel("")
        ax4b.set_ylabel("Рейтинг (0-10)")
        ax4b.tick_params(axis="x", rotation=0)
        st.pyplot(fig4b)
        plt.close(fig4b)

    win, other = oscar_stats.loc[True], oscar_stats.loc[False]
    st.caption(
        f"У вибірці переможців «Оскара» за найкращий фільм: {n_winners}, інших фільмів: {len(fdf) - n_winners}. "
        f"Медіанні значення для переможців: бюджет {fmt(win['budget'] / 1_000_000, 1)} млн $, "
        f"збори {fmt(win['gross'] / 1_000_000, 1)} млн $, рейтинг {fmt(win['score'], 1)}. "
        f"Для решти фільмів: бюджет {fmt(other['budget'] / 1_000_000, 1)} млн $, "
        f"збори {fmt(other['gross'] / 1_000_000, 1)} млн $, рейтинг {fmt(other['score'], 1)}. "
        "Переможців небагато, тому висновок орієнтовний, а це лише зв'язок, а не доказ впливу нагороди. "
        "Позначку про «Оскар» додано вручну з Вікіпедії (список переможців у категорії «Найкращий фільм»)."
    )

# Висновки
st.header("Висновки (за всім датасетом)")
st.markdown(
    """
1. Окупність фільмів з часом зросла. Медіанна окупність спочатку впала з 1,57 у 1980-х до 1,15 у 1990-х, а потім зросла до 1,88 у 2000-х і 2,54 у 2010-х.
2. Довші фільми мають вищий середній рейтинг. Короткі фільми (до 90 хв): 5,90, середні (91-120 хв): 6,32, довгі (понад 120 хв): 7,04. Це лише зв'язок, а не доказ впливу тривалості на оцінку.
3. Переможці «Оскара» значно успішніші за зборами та рейтингом, але не за бюджетом. Медіанний бюджет майже однаковий (20 і 20,5 млн $), проте збори переможців приблизно в 11 разів більші (222 млн $ проти 20 млн $), а рейтинг вищий (8,0 проти 6,5). Переможців лише 40, тому висновок орієнтовний.
4. Бюджет і збори помітно пов'язані. Коефіцієнт кореляції 0,74: дорожчі фільми загалом збирають більше, але високий бюджет не гарантує успіху.
"""
)
