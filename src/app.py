# backend/app.py
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Apply CORS to all routes

# Define weights for each statistic (adjust as needed)
weights = {
    'SCA SCA': 1,
    'SCA SCA90': 1,
    'SCA Types PassLive': 0.5,
    'SCA Types PassDead': 0.5,
    'SCA Types TO': 0.5,
    'SCA Types Sh': 0.5,
    'SCA Types Fld': 0.5,
    'SCA Types Def': 0.5,
    'GCA GCA': 1,
    'GCA GCA90': 1,
    'GCA Types PassLive': 0.5,
    'GCA Types PassDead': 0.5,
    'GCA Types TO': 0.5,
    'GCA Types Sh': 0.5,
    'GCA Types Fld': 0.5,
    'GCA Types Def': 0.5,
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

# Convert all columns to numeric type
for col in df.columns:
    if col not in ['Player', 'Nation', 'Pos', 'Squad', 'Comp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate aggregated score for each player
score_columns = [col for col in df.columns if 'SCA' in col or 'GCA' in col]
df['score'] = df[score_columns].dot(pd.Series(weights))

# Convert 'score' column to numeric type
df['score'] = pd.to_numeric(df['score'], errors='coerce')


@app.route('/players')
def get_players():
    sorted_players = df.sort_values(by='score', ascending=False)[['Player', 'score']]
    players_json = sorted_players.to_dict(orient='records')
    return jsonify(players_json)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
