import streamlit as st
import preprocessor
import helper
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')

st.set_page_config(page_title="WhatsApp Chat and Sentiment Analyzer", layout="wide")

# Sidebar
st.sidebar.title('WhatsApp Chat and Sentiment Analyzer')
st.sidebar.markdown("Upload your exported chat file below to begin analysis.")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=[".txt"])

# Main logic
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    if df.empty:
        st.error("⚠️ No messages could be parsed from the uploaded file. Please make sure the file is a valid exported WhatsApp chat (with timestamps) and is not empty.")
    else:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

        # User Selection
        user_list = df['user'].unique().tolist()
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Analyze chat for:", user_list)

        if st.sidebar.button("Show Analysis"):

            # Top Statistics
            st.header('Top Statistics')
            num_messages, words, num_media_messages, links = helper.fetch_stats(selected_user, df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", num_messages)
            with col2:
                st.metric("Total Words", words)
            with col3:
                st.metric("Media Shared", num_media_messages)
            with col4:
                st.metric("Links Shared", links)

            # Most Busy Users (Only for Overall)
            if selected_user == "Overall":
                st.header("Most Active Users")
                x, new_df = helper.most_busy_user(df)
                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = pd.DataFrame({
                        'User': x.index,
                        'Messages': x.values
                    })
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('User:N', sort='-y', title="User"),
                        y=alt.Y('Messages:Q', title="Number of Messages"),
                        color=alt.value('#25D366'), # WhatsApp Green
                        tooltip=['User', 'Messages']
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                with col2:
                    st.dataframe(new_df, use_container_width=True)

            # Word Cloud
            st.header('☁️ Word Cloud')
            df_wc = helper.create_word_cloud(selected_user, df)
            if df_wc is not None:
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("ℹ️ Not enough text messages to generate a word cloud (or all messages were media / notifications).")

            # Emoji Analysis
            st.header('Emoji Usage')
            emoji_df = helper.emoji_helper(selected_user, df)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(emoji_df, use_container_width=True)
            with col2:
                if not emoji_df.empty:
                    top_emojis = emoji_df.head(10)
                    chart = alt.Chart(top_emojis).mark_bar().encode(
                        x=alt.X('emoji:N', sort='-y', title="Emoji", axis=alt.Axis(labelAngle=0, labelFontSize=16)),
                        y=alt.Y('count:Q', title="Frequency"),
                        color=alt.value('#128C7E'), # WhatsApp Teal
                        tooltip=['emoji', 'count']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No emojis used in this chat.")

            # Monthly Timeline
            st.header("📆 Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            chart = alt.Chart(timeline).mark_line(color='#25D366', point=True).encode(
                x=alt.X('time:N', sort=None, title="Month-Year"),
                y=alt.Y('message:Q', title="Messages"),
                tooltip=['time', 'message']
            ).properties(height=350).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Daily Timeline
            st.header("🗓️ Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            daily_timeline['only_date'] = pd.to_datetime(daily_timeline['only_date'])
            chart = alt.Chart(daily_timeline).mark_line(color='#128C7E').encode(
                x=alt.X('only_date:T', title="Date"),
                y=alt.Y('message:Q', title="Messages"),
                tooltip=[alt.Tooltip('only_date:T', title='Date', format='%Y-%m-%d'), 'message']
            ).properties(height=350).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Activity Map
            st.header('Activity Map')
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Most Active Days')
                busy_day_series = helper.week_activity_map(selected_user, df)
                busy_day = pd.DataFrame({
                    'Day': busy_day_series.index,
                    'Messages': busy_day_series.values
                })
                chart = alt.Chart(busy_day).mark_bar().encode(
                    x=alt.X('Day:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], title="Day of Week"),
                    y=alt.Y('Messages:Q', title="Messages"),
                    color=alt.value('#128C7E'),
                    tooltip=['Day', 'Messages']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            with col2:
                st.subheader('Most Active Months')
                busy_month_series = helper.month_activity_map(selected_user, df)
                busy_month = pd.DataFrame({
                    'Month': busy_month_series.index,
                    'Messages': busy_month_series.values
                })
                chart = alt.Chart(busy_month).mark_bar().encode(
                    x=alt.X('Month:N', sort=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], title="Month"),
                    y=alt.Y('Messages:Q', title="Messages"),
                    color=alt.value('#25D366'),
                    tooltip=['Month', 'Messages']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # Heat Map
            st.header('Heat Map')
            user_heatmap = helper.activity_heat_map(selected_user, df)
            heatmap_df = user_heatmap.reset_index().melt(id_vars='day_name', var_name='period', value_name='messages')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            chart = alt.Chart(heatmap_df).mark_rect().encode(
                x=alt.X('period:O', title="Hour Period"),
                y=alt.Y('day_name:O', sort=day_order, title="Day"),
                color=alt.Color('messages:Q', scale=alt.Scale(scheme='greens'), title="Messages"),
                tooltip=['day_name', 'period', 'messages']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Sentiment Analysis
            st.header("Sentiment Analysis")
            total_positive, total_negative, total_neutral = helper.sentiment_analysis(selected_user, df)
            total_msgs = total_positive + total_negative + total_neutral
            if total_msgs == 0:
                st.info("No valid messages for sentiment analysis.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{(total_positive / total_msgs) * 100:.2f}%")
                with col2:
                    st.metric("Negative", f"{(total_negative / total_msgs) * 100:.2f}%")
                with col3:
                    st.metric("Neutral", f"{(total_neutral / total_msgs) * 100:.2f}%")

                st.subheader("Sentiment Distribution")
                sentiment_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Messages": [total_positive, total_negative, total_neutral]
                })
                chart = alt.Chart(sentiment_data).mark_arc(innerRadius=60, outerRadius=100).encode(
                    theta=alt.Theta(field="Messages", type="quantitative"),
                    color=alt.Color(field="Sentiment", type="nominal", scale=alt.Scale(
                        domain=["Positive", "Negative", "Neutral"],
                        range=["#25D366", "#E53935", "#B0BEC5"]
                    )),
                    tooltip=["Sentiment", "Messages"]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # ══════════════════════════════════════════════
            #  SECTION A — RESPONSE TIME ANALYSIS
            # ══════════════════════════════════════════════

            st.markdown("---")
            st.title("⏱️ Response Time Analysis")

            rt = helper.response_time_analysis(df, selected_user)

            avg_rt   = rt['avg_response_times']
            initiators = rt['initiator_counts']
            conv_df  = rt['conv_df']

            if avg_rt.empty:
                st.info("Not enough data to compute response times for this selection.")
            else:
                # ── Row 1 : Metric cards ──────────────────────────────────────────────
                col1, col2, col3 = st.columns(3)

                fastest_user = avg_rt.idxmin()
                fastest_time = avg_rt.min()
                total_convs  = len(conv_df)
                avg_conv_len = conv_df['message_count'].mean()

                col1.metric("Fastest Responder",  f"{fastest_user}",       f"{fastest_time:.1f} min avg")
                col2.metric("Total Conversations", f"{total_convs}")
                col3.metric("Avg Messages / Conv", f"{avg_conv_len:.1f}")

                # ── Row 2 : Bar chart – avg response time per user ───────────────────
                st.subheader("Average Response Time per User (minutes)")
                st.caption("Lower = faster responder. Capped at 240 min to exclude overnight gaps.")

                rt_df = pd.DataFrame({
                    'User': avg_rt.index,
                    'Avg_Response_Minutes': avg_rt.values
                }).sort_values(by='Avg_Response_Minutes', ascending=True)

                rt_chart = (
                    alt.Chart(rt_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X('User:N', sort=None, axis=alt.Axis(labelAngle=-30)),
                        y=alt.Y('Avg_Response_Minutes:Q', title='Minutes'),
                        tooltip=['User', 'Avg_Response_Minutes'],
                        color=alt.value('#25D366')  # WhatsApp Green
                    )
                    .properties(height=320)
                )
                st.altair_chart(rt_chart, use_container_width=True)

                # ── Row 3 : Conversation initiators ──────────────────────────────────
                st.subheader("Who Starts Conversations?")
                st.caption("Counted whenever a message arrives after a 4-hour silence gap.")

                init_df = pd.DataFrame({
                    'User': initiators.index,
                    'Conversations_Started': initiators.values
                }).sort_values(by='Conversations_Started', ascending=False)

                init_chart = (
                    alt.Chart(init_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X('User:N', sort=None, axis=alt.Axis(labelAngle=-30)),
                        y=alt.Y('Conversations_Started:Q', title='Conversations Started'),
                        tooltip=['User', 'Conversations_Started'],
                        color=alt.value('#128C7E')  # WhatsApp Teal
                    )
                    .properties(height=300)
                )
                st.altair_chart(init_chart, use_container_width=True)

                # ── Row 4 : Conversation length distribution ──────────────────────────
                st.subheader("Conversation Length Distribution")
                st.caption("How many messages long are typical conversations?")

                hist_chart = (
                    alt.Chart(conv_df)
                    .mark_bar(opacity=0.8)
                    .encode(
                        x=alt.X('message_count:Q', bin=alt.Bin(maxbins=25), title='Messages in Conversation'),
                        y=alt.Y('count():Q', title='Number of Conversations'),
                        tooltip=['count()']
                    )
                    .properties(height=280)
                    .interactive()
                )
                st.altair_chart(hist_chart, use_container_width=True)


            # ══════════════════════════════════════════════
            #  SECTION B — TOPIC MODELING
            # ══════════════════════════════════════════════

            st.markdown("---")
            st.title("🧠 Topic Modeling")
            st.caption(
                "Uses LDA (Latent Dirichlet Allocation) to automatically discover the main "
                "themes your group talks about."
            )

            num_topics = st.slider("Number of topics to discover", min_value=3, max_value=10, value=5)

            with st.spinner("Running topic model… this may take a few seconds."):
                topics, topic_labels = helper.get_topics(df, selected_user, num_topics=num_topics)

            if not topics:
                st.warning("Not enough text data to run topic modeling. Try selecting 'Overall' or a longer chat.")
            else:
                # Show each topic as an expandable card with a word-weight bar chart
                for (topic_id, word_weights), label in zip(topics, topic_labels):
                    with st.expander(label, expanded=(topic_id == 0)):
                        words   = [w for w, _ in word_weights]
                        weights = [round(float(wt), 4) for _, wt in word_weights]

                        topic_df = pd.DataFrame({'Word': words, 'Weight': weights})

                        topic_chart = (
                            alt.Chart(topic_df)
                            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                            .encode(
                                x=alt.X('Weight:Q', title='Relevance Score'),
                                y=alt.Y('Word:N', sort='-x'),
                                tooltip=['Word', 'Weight'],
                                color=alt.value('#7F77DD')  # purple
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(topic_chart, use_container_width=True)

                # Summary table of all topics
                st.subheader("All Topics at a Glance")
                summary_rows = []
                for (topic_id, word_weights), label in zip(topics, topic_labels):
                    top_words = ', '.join([w for w, _ in word_weights[:5]])
                    summary_rows.append({'Topic': f"Topic {topic_id + 1}", 'Top Words': top_words})

                st.table(pd.DataFrame(summary_rows))

