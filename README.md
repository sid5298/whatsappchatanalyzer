# 📱 WhatsApp Chat & Sentiment Analyzer

An interactive, premium **Streamlit** web application that parses exported WhatsApp chat logs (`.txt` files) and extracts deep analytical insights. This dashboard visualizes conversation dynamics, user activity, messaging timelines, emoji usage, sentiment levels, and features advanced data science implementations like **Response Time Analysis** and **Topic Modeling (LDA)**.

👉 **Perfect for data scientist & analyst portfolios!**

---

## ✨ Features

### 📊 Basic & Intermediate Analytics
* **Top Statistics:** Instantly view total messages, word count, media count, and shared links.
* **Most Active Users:** Highlights the top contributors in the group with interactive bar charts and percentage breakdowns.
* **Word Cloud:** Generates a clean, filter-enhanced word cloud of the most frequently used words.
* **Emoji Usage Insights:** Identifies, counts, and graphs the top 10 emojis used in the chat.
* **Monthly & Daily Timelines:** Interactive line charts representing chat activity trends over years, months, and days.
* **Activity Maps:** Identifies the busiest days of the week and months of the year.
* **Weekly Activity Heatmap:** An interactive grid visualizer showing which hours of the week are the most active.

### 🧠 Advanced Data Science Additions
* **Sentiment Analysis:** Evaluates every message using NLTK's VADER Sentiment Intensity Analyzer, categorizing chat vibe into positive, negative, and neutral metrics.
* **Response Time Analysis:**
  * Calculates the **Average Response Time (reply latency)** per user in minutes.
  * Tracks **Conversation Initiators** (who starts conversations after a 4-hour silence gap).
  * Charts **Conversation Length Distribution** (how many messages a typical conversation lasts).
* **Topic Modeling (LDA):**
  * Applies **Latent Dirichlet Allocation (unsupervised machine learning)** via `gensim` to dynamically group chat discussions into thematic clusters.
  * Filters out custom regional and domain-specific stopwords.
  * Generates interactive, weight-based bar charts showing the relevance score of keywords for each topic.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python 3.12+
* **Dashboard Framework:** Streamlit
* **Data Processing:** Pandas, Regex (`re`)
* **Visualization:** Altair (modern, interactive Vega-Lite graphics)
* **Natural Language Processing (NLP):** NLTK (VADER Sentiment), WordCloud
* **Machine Learning:** Gensim (LDA Topic Modeling)
* **Utility Libraries:** Emoji, URLExtract

---

## 🚀 Running the Project Locally

### 1. Clone the repository:
```bash
git clone https://github.com/sid5298/whatsappchatanalyzer.git
cd whatsappchatanalyzer
```

### 2. Install dependencies:
```bash
pip install streamlit pandas altair matplotlib seaborn nltk gensim pyLDAvis scikit-learn emoji urlextract wordcloud
```

### 3. Run the Streamlit app:
```bash
streamlit run app.py
```

### 4. How to analyze your chats:
1. Open WhatsApp on your phone and open a chat.
2. Tap **Settings (three dots) ➔ More ➔ Export Chat**.
3. Select **Without Media** (this yields a `.txt` file).
4. Upload that `.txt` file into the sidebar of the web app and click **Show Analysis**!
