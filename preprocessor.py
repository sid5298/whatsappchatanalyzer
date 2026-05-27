import pandas as pd
import re

def preprocess(data):
    patterns = [
        # Original format: dd/mm/yy, hh:mm -
        r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        # Format with AM/PM: dd/mm/yy, hh:mm am - or hh:mm PM -
        r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[aApPmM]{2}\s-\s',
        # iOS format with brackets: [dd/mm/yy, hh:mm:ss]
        r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:[aApPmM]{2})?\]\s',
        # Hyphenated format: dd-mm-yyyy, hh:mm -
        r'\d{1,2}-\d{1,2}-\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        # ISO format: yyyy-mm-dd, hh:mm -
        r'\d{4}-\d{2}-\d{2},\s\d{2}:\d{2}\s-\s'
    ]

    # Find the pattern that matches the most dates
    pattern = patterns[0]
    max_matches = 0
    for p in patterns:
        matches = len(re.findall(p, data))
        if matches > max_matches:
            max_matches = matches
            pattern = p

    msgs = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': msgs, 'message_date': dates})

    # If no messages matched the pattern, return an empty dataframe with correct schema
    if df.empty:
        df = pd.DataFrame(columns=['user_message', 'message_date', 'date', 'user', 'message', 'year', 'month', 'day', 'hour', 'minute', 'only_date', 'day_name', 'period'])
        df['user'] = df['user'].astype(str)
        df['message'] = df['message'].astype(str)
        return df

    # Clean the date strings to remove brackets and trailing hyphens/spaces
    df['message_date'] = df['message_date'].apply(lambda x: x.replace('[', '').replace(']', '').strip(' -').strip())
    
    df['message_date'] = pd.to_datetime(df['message_date'], dayfirst=True, errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages

    # Ensure user and message columns are string type
    df['user'] = df['user'].astype(str)
    df['message'] = df['message'].astype(str)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()

    df['period'] = df['date'].dt.hour.apply(lambda x: f'{int(x):02d}-{(int(x) + 1) % 24:02d}' if pd.notna(x) else '00-00')

    return df


















