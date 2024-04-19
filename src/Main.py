# libraries
import pandas as pd

# Define weights for each statistic (adjust as needed)
weights = {
    'SCA': 1,
    # Add other statistics and their weights here
}

# fbref table link
url_df = 'https://fbref.com/en/comps/Big5/2022-2023/gca/players/2022-2023-Big-5-European-Leagues-Stats'

# Read HTML tables from the URL
df = pd.read_html(url_df)[0]

df.columns = [' '.join(col).strip() for col in df.columns]

df = df.reset_index(drop=True)

# creating a list with new names
new_columns = []
for col in df.columns:
    if 'level_0' in col:
        new_col = col.split()[-1]  # takes the last name
    else:
        new_col = col
    new_columns.append(new_col)

# rename columns
df.columns = new_columns
df = df.fillna(0)

# Calculate aggregated score for each player
df['score'] = (df['SCA SCA'] * weights['SCA'])

# Convert 'score' column to numeric type
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# Sort players by score
sorted_players = df.sort_values(by='score', ascending=False)['Player']

# Print sorted player names
print(sorted_players)
