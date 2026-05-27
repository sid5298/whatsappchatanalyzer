import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_user(df):
    x = df['user'].value_counts().head()
    percent_series = round((df['user'].value_counts() / df.shape[0]) * 100, 2)
    new_df = pd.DataFrame({
        'Name': percent_series.index,
        'Percent': percent_series.values
    })
    return x, new_df

def create_word_cloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out unwanted messages
    filtered_df = df[
        (~df['message'].str.contains('<Media omitted>', na=False)) &
        (df['message'].str.lower() != 'null') &
        (df['user'] != 'group_notification') &
        (df['message'].str.strip() != '')
    ]

    if filtered_df.empty or filtered_df['message'].str.cat(sep=" ").strip() == "":
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')
    df_wc = wc.generate(filtered_df['message'].str.cat(sep=" "))
    return df_wc


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heat_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out media messages
    df = df[~df['message'].str.contains('<Media omitted>', na=False)]

    sentiments = SentimentIntensityAnalyzer()

    total_positive = 0
    total_negative = 0
    total_neutral = 0
    total_messages = 0

    for msg in df['message']:
        score = sentiments.polarity_scores(msg)
        total_messages += 1
        if score['compound'] >= 0.05:
            total_positive += 1
        elif score['compound'] <= -0.05:
            total_negative += 1
        else:
            total_neutral += 1

    return total_positive, total_negative, total_neutral


# ─────────────────────────────────────────────
#  FEATURE 1 — RESPONSE TIME ANALYSIS
# ─────────────────────────────────────────────

def response_time_analysis(df, selected_user):
    """
    Returns a dict with:
      - avg_response_times : Series  — avg reply latency per user (minutes)
      - initiator_counts   : Series  — how many conversations each user started
      - conv_df            : DataFrame — raw conversation-level data for charting
    """
    # Work on a copy; drop system notifications
    data = df[df['message'] != '<Media omitted>'].copy()
    data = data[~data['message'].str.startswith("Messages and calls are end-to-end encrypted")]
    data = data.sort_values('date').reset_index(drop=True)

    # ── Conversation segmentation ──────────────────────────────────────────
    # A new conversation starts when there is a gap of > 4 hours since the
    # last message (the same threshold WhatsApp uses for "today/yesterday").
    CONV_GAP = pd.Timedelta(hours=4)
    data['time_diff'] = data['date'].diff()
    data['new_conv']  = (data['time_diff'] > CONV_GAP) | (data['time_diff'].isna())
    data['conv_id']   = data['new_conv'].cumsum()

    # ── Response latency ───────────────────────────────────────────────────
    # For every consecutive pair of messages WITHIN a conversation where the
    # sender changes, compute the gap.  That gap is a "response time".
    data['prev_user'] = data['user'].shift(1)
    data['prev_time'] = data['date'].shift(1)
    data['prev_conv'] = data['conv_id'].shift(1)

    responses = data[
        (data['user'] != data['prev_user']) &   # sender changed
        (data['conv_id'] == data['prev_conv'])   # same conversation
    ].copy()

    responses['response_minutes'] = (
        (responses['date'] - responses['prev_time'])
        .dt.total_seconds() / 60
    )

    # Cap extreme outliers at 240 min (4 h) so the average stays meaningful
    responses['response_minutes'] = responses['response_minutes'].clip(upper=240)

    if selected_user != 'Overall':
        responses = responses[responses['user'] == selected_user]

    avg_response_times = (
        responses.groupby('user')['response_minutes']
        .mean()
        .round(1)
        .sort_values()
    )

    # ── Conversation initiators ────────────────────────────────────────────
    initiators = data[data['new_conv']].copy()
    if selected_user != 'Overall':
        initiators = initiators[initiators['user'] == selected_user]

    initiator_counts = (
        initiators.groupby('user')
        .size()
        .sort_values(ascending=False)
    )

    # ── Conversation-level DataFrame (for optional deep-dive charts) ───────
    conv_df = (
        data.groupby('conv_id')
        .agg(
            start_time=('date', 'min'),
            end_time=('date', 'max'),
            message_count=('message', 'count'),
            participants=('user', lambda x: x.nunique())
        )
        .reset_index()
    )
    conv_df['duration_minutes'] = (
        (conv_df['end_time'] - conv_df['start_time'])
        .dt.total_seconds() / 60
    ).round(1)

    return {
        'avg_response_times': avg_response_times,
        'initiator_counts':   initiator_counts,
        'conv_df':            conv_df,
    }


# ─────────────────────────────────────────────
#  FEATURE 2 — TOPIC MODELING (LDA)
# ─────────────────────────────────────────────

def get_topics(df, selected_user, num_topics=5, num_words=8):
    """
    Runs LDA topic modeling on the chat messages.

    Returns:
      - topics : list of (topic_id, [(word, weight), ...])
      - topic_labels : list of str — short human-readable labels per topic
    """
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from gensim.parsing.preprocessing import STOPWORDS
    import re

    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    # ── Text cleaning ──────────────────────────────────────────────────────
    EXTRA_STOPWORDS = {
        'media', 'omitted', 'message', 'deleted', 'http', 'https',
        'www', 'yeah', 'okay', 'ok', 'yes', 'no', 'hai', 'haa',
        'naa', 'ani', 'undi', 'ila', 'ante', 'kuda', 'ga', 'lo',  # Telugu common words
        'ha', 'haha', 'lol', 'bro', 'yep', 'nope', 'oh', 'hi', 'hey'
    }
    all_stopwords = STOPWORDS.union(EXTRA_STOPWORDS)

    def clean(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)          # remove URLs
        text = re.sub(r'[^a-z\s]', '', text)         # keep only letters
        tokens = text.split()
        tokens = [t for t in tokens if t not in all_stopwords and len(t) > 3]
        return tokens

    data = data[data['message'] != '<Media omitted>']
    data = data[~data['message'].str.startswith("Messages and calls")]

    corpus_tokens = data['message'].apply(clean).tolist()
    corpus_tokens = [t for t in corpus_tokens if len(t) > 0]  # drop empty docs

    if len(corpus_tokens) < 20:
        return [], []   # not enough data

    # ── Build dictionary & bag-of-words ───────────────────────────────────
    dictionary = corpora.Dictionary(corpus_tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.6)  # remove very rare/common
    bow_corpus  = [dictionary.doc2bow(doc) for doc in corpus_tokens]
    bow_corpus  = [d for d in bow_corpus if len(d) > 0]

    if len(bow_corpus) < 10:
        return [], []

    # ── Train LDA ─────────────────────────────────────────────────────────
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    topics = lda_model.show_topics(
        num_topics=num_topics,
        num_words=num_words,
        formatted=False
    )

    # ── Auto-label each topic with its top 3 words ─────────────────────────
    topic_labels = []
    for topic_id, word_weights in topics:
        top_words = [w for w, _ in word_weights[:3]]
        topic_labels.append(f"Topic {topic_id + 1}: {', '.join(top_words)}")

    return topics, topic_labels



