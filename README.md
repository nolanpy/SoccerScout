# SoccerScout

SoccerScout is a tool for soccer clubs to identify undervalued and overvalued players by analyzing advanced statistics and comparing predicted values to actual market values.

## Next Steps

- [ ] Add ML models to predict market values
- [ ] Compare predicted vs. actual values
- [ ] Track prediction accuracy over time
- [ ] Add more visualization tools

## Overview

This application analyzes player performance using a comprehensive set of advanced statistics including:

- Expected goals (xG) and assists (xA)
- Shot-creating actions (SCA) and goal-creating actions (GCA)
- Progressive passes and carries
- Defensive metrics like tackles, interceptions, and pressures
- Physical and tactical data

The current version uses simulated player data in a SQLite database, with plans to integrate with real-world APIs and implement machine learning models.

## Features

- **Player Performance Scoring**: Calculate weighted performance scores based on 30+ advanced metrics
- **Market Value Comparison**: Compare performance to market value to identify undervalued/overvalued players
- **Interactive UI**: Filter and search for players, view detailed statistics
- **Data Visualization**: View statistical distributions across metrics

## Tech Stack

- **Backend**: Python, Flask, SQLite, Pandas
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Data**: Simulated soccer statistics database with 20+ players and 3 seasons of data

## Machine Learning Implementation

SoccerScout now includes a comprehensive machine learning system to predict player market values and identify undervalued talent:

1. **Position-Specific Models**: 
   - Separate models for forwards, midfielders, defenders, and goalkeepers
   - Position-appropriate weighting of statistics (offensive vs. defensive)
   - Accounts for position-specific market dynamics (forwards typically cost more)
   - Falls back to general model when position-specific data is limited

2. **Age-Adjusted Evaluation**:
   - Incorporates player age into market value predictions
   - Models peak value curve (typically around age 27)
   - Boosts values for young players showing high performance (potential)
   - Accounts for declining values in older players
   - Enables meaningful comparisons between players of different ages

3. **Multiple ML Algorithms**: 
   - Random Forest Regressor (default)
   - Gradient Boosting Regressor
   - Ridge Regression
   - Lasso Regression

4. **Historical Data Analysis**:
   - Uses 6 seasons of data (2018-2023) to train models
   - Predicts current season (2023-2024) market values
   - Evaluates prediction accuracy season-by-season
   - Improves models by learning from historical trends

5. **Feature Importance Analysis**:
   - Identifies which statistics most strongly correlate with market value
   - Automatically adjusts statistical weights based on ML insights
   - Generates optimal weights for the player evaluation algorithm
   - Tailors importance by position (goals matter more for forwards, tackles for defenders)

6. **Model Performance Metrics**:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score (coefficient of determination)
   - Season-by-season trend analysis

## Development Goals

1. ✅ Implement ML model to predict player market values
2. ✅ Use historical data to validate predictions
3. Add more visualization tools and a dedicated ML dashboard
4. Incorporate real-world data from soccer APIs

## Dev Guidelines

- Push any changes to the `development` branch and open a PR
- Add bugs and features to Issues

## Setup and Running

1. Install dependencies:

   ```
   pip install flask flask_cors pandas
   ```

2. Run the application:

   ```
   cd src
   python app.py
   ```

3. Access the UI in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## API Endpoints

### Core Endpoints
- `/players` - Get all players with scores and stats
- `/player/<id>` - Get detailed stats for a specific player
- `/top-undervalued` - Get top 10 undervalued players
- `/stats-distribution` - Get statistical distributions for key metrics

### Machine Learning Endpoints
- `/ml-predictions` - Get market value predictions with position and age adjustments
  - Query params: `position_specific` (true/false), `age_adjusted` (true/false)
- `/compare-players` - Compare players within the same position and/or age group
  - Query params: `position` (e.g., 'CF' or 'forward'), `age_group` (e.g., 'youth', 'prime'), `player_id` (to specify a reference player)
- `/train-model` - Train the ML model and get performance metrics
- `/compare-models` - Train and compare different ML models (Random Forest, Gradient Boosting, Ridge, Lasso)
- `/update-weights` - Update statistical weights based on ML model feature importance

## Potential Future Data Sources:

Paid:

- Wyscout https://www.hudl.com/products/wyscout
- Opta Sports API

Free:

- FBref
- Understat
- Whoscored

## Resources:

- [Scraping FBref with Python](https://medium.com/@ricardoandreom/how-to-scrape-and-personalize-data-from-fbref-with-python-a-guide-to-unlocking-football-insights-7e623607afca)
- [Soccer API Scraping Guide](https://ctomashot.medium.com/how-i-scraped-an-api-for-my-soccer-scouting-app-c67df68da6ca)
- [Machine Learning Algorithms Overview](https://builtin.com/data-science/tour-top-10-algorithms-machine-learning-newbies)
