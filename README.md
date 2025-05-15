# SoccerScout

SoccerScout is an advanced soccer player analysis and scouting tool that combines data science, machine learning, and web scraping to identify undervalued talent in the soccer market. The application provides comprehensive player statistics, performance metrics, and market value predictions to support recruitment decisions.

## Next Steps

- [x] Add ML models to predict market values
- [x] Compare predicted vs. actual values
- [x] Track prediction accuracy over time
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

1. **Unified ML Model**:

   - Centralized machine learning framework in `unified_ml_model.py`
   - Combines multiple prediction approaches in a single module
   - Provides consistent API for training, evaluation, and prediction
   - Handles data preprocessing and feature engineering automatically

2. **Position-Specific Models**:

   - Separate models for forwards, midfielders, defenders, and goalkeepers
   - Position-appropriate weighting of statistics (offensive vs. defensive)
   - Accounts for position-specific market dynamics (forwards typically cost more)
   - Falls back to general model when position-specific data is limited

3. **Age-Adjusted Evaluation**:

   - Incorporates player age into market value predictions
   - Models peak value curve (typically around age 27)
   - Boosts values for young players showing high performance (potential)
   - Accounts for declining values in older players
   - Enables meaningful comparisons between players of different ages

4. **Multiple ML Algorithms**:

   - Random Forest Regressor (default)
   - Gradient Boosting Regressor
   - Ridge Regression
   - Lasso Regression

5. **Historical Data Analysis**:

   - Uses 6 seasons of data (2018-2023) to train models
   - Predicts current season (2023-2024) market values
   - Evaluates prediction accuracy season-by-season
   - Improves models by learning from historical trends

6. **Feature Importance Analysis**:

   - Identifies which statistics most strongly correlate with market value
   - Automatically adjusts statistical weights based on ML insights
   - Generates optimal weights for the player evaluation algorithm
   - Tailors importance by position (goals matter more for forwards, tackles for defenders)

7. **Value Trajectory Prediction**:

   - Projects player values into future seasons
   - Considers age progression and position-specific career curves
   - Identifies players with high growth potential
   - Helps identify future investment opportunities

8. **Model Performance Metrics**:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score (coefficient of determination)
   - Season-by-season trend analysis

## ML Metrics Explained

### R² Score (Coefficient of Determination)

- **What it means**: R² measures how well the model's predictions explain the variation in the actual player market values.
- **Interpretation**: Ranges from 0 to 1, where 1 means perfect prediction.
- **Practical value**: Higher R² indicates the model better captures what determines a player's market value.

### RMSE (Root Mean Squared Error)

- **What it means**: The square root of the average squared difference between predicted and actual market values.
- **Interpretation**: Lower is better; shows the average magnitude of prediction errors in the same units as market value (euros).
- **Practical value**: Helps scouts understand the expected margin of error in predictions.

### MAE (Mean Absolute Error)

- **What it means**: The average absolute difference between predicted and actual market values.
- **Interpretation**: Lower is better; more intuitive than RMSE as it's a direct average of errors.
- **Practical value**: Easier for non-technical users to understand - "on average, predictions are off by €X million".

### Percentage Error

- **What it means**: The average percentage by which predictions differ from actual values.
- **Interpretation**: Lower is better; helpful for comparing errors across players with different market values.
- **Practical value**: Helps identify whether predictions are more accurate for high or low-value players.

### Value Ratio

- **What it means**: The ratio of predicted value to actual market value.
- **Interpretation**:
  - Ratio > 1.5: Player is undervalued (worth more than market price)
  - 0.7 <= Ratio <= 1.5: Fair value
  - Ratio < 0.7: Player is overvalued (worth less than market price)
- **Practical value**: The core metric for talent scouts to identify market inefficiencies and potential bargains.

### Feature Importance

- **What it means**: Indicates which player statistics most heavily influence market value predictions.
- **Interpretation**: Higher values indicate greater impact on player valuation.
- **Practical value**: Helps scouts understand which skills and statistics truly drive player market values.

## Performance Tracking and Comparison

The system now includes comprehensive tools to track and compare model performance across different datasets and configurations:

1. **Performance Logging**

   - Detailed logs of training process and metrics
   - Standardized metrics format for easy comparison
   - Timestamps and configuration details preserved

2. **Metrics Storage**

   - JSON-based storage of model metrics
   - Organized by model type and custom tags
   - Preserves data characteristics for context

3. **Run Tagging**

   - Tag each training run (e.g., "baseline", "more_data", "real_data")
   - Compare performance between specific tagged runs
   - Track progress over time with sequential tags

4. **Before/After Comparison**
   - Direct comparison of key metrics (R², RMSE, MAE)
   - Percentage improvements calculated automatically
   - Overall assessment of model improvement

### How to Track Performance When Adding Data

1. First, establish a baseline:

   ```
   http://localhost:5000/train-model?tag=baseline
   ```

2. After adding more data, create a new tagged run:

   ```
   http://localhost:5000/train-model?tag=more_data
   ```

3. Compare the performance:

   ```
   http://localhost:5000/compare-model-runs?baseline_tag=baseline&comparison_tag=more_data
   ```

4. Analyze the metrics to see how your data additions impacted performance

## Time-Series Value Prediction

SoccerScout currently implements season-by-season analytics comparison to predict future values by:

1. Training models on historical seasons (2018-2023) to learn how statistics correlate with market values in those specific seasons
2. Using the dedicated value trajectory model (`train_value_trajectory_model` method) that analyzes growth patterns across seasons
3. Calculating 1-year and 3-year value growth metrics to identify trends
4. Applying position-specific career trajectories with different peak ages and growth curves
5. Projecting future values through the `predict_future_values` method that simulates season-by-season progression
6. Incorporating age progression and expected statistical performance evolution

This approach allows the system to understand how a player's statistical profile in a given season translates to market value, and then apply that knowledge to predict future values based on current statistics.

## Development Goals

1. ✅ Implement ML model to predict player market values
2. ✅ Use historical data to validate predictions
3. ✅ Add performance tracking and comparison tools
4. ✅ Develop unified ML model with position-specific and age-adjusted predictions
5. ✅ Implement value trajectory prediction for future player values
6. Add more visualization tools and a dedicated ML dashboard
7. Incorporate additional real-world data from soccer APIs
8. Enhance the season-to-season statistical improvement analysis to better predict value changes
9. Develop a "what-if" simulator to project how specific stat improvements would affect a player's market value

## Dev Guidelines

- Push any changes to the `development` branch and open a PR
- Add bugs and features to Issues

## Setup and Running

1. Install dependencies:

   ```
   pip install flask flask_cors pandas numpy scikit-learn beautifulsoup4 requests
   ```

2. Run the application:

   ```
   cd src
   python app.py
   ```

3. Access the UI in your browser:
   ```
   http://127.0.0.1:5000/players
   ```

4. Development commands:
   ```
   # Lint code
   pip install flake8 && flake8 src/

   # Format code
   pip install black && black src/
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
- `/transfer-value-predictions` - Get detailed transfer value analysis with filtering
  - Query params: `status` (undervalued/overvalued/fair), `position` (e.g., 'CF')
- `/analyze-transfer-values` - Run a full transfer value analysis to find undervalued and overvalued players
  - Query params: `force` (true/false) - Force reanalysis even if recent results exist
- `/compare-players` - Compare players within the same position and/or age group
  - Query params: `position` (e.g., 'CF' or 'forward'), `age_group` (e.g., 'youth', 'prime'), `player_id` (to specify a reference player)
- `/train-model` - Train the ML model and get performance metrics
  - Query params:
    - `tag` (e.g., 'baseline', 'more_data') - Tag to identify this training run
    - `model_type` ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
    - `position_specific` (true/false)
    - `age_adjusted` (true/false)
    - `time_series` (true/false) - Whether to include time-series features
- `/compare-models` - Train and compare different ML models (Random Forest, Gradient Boosting, Ridge, Lasso)
- `/compare-model-runs` - Compare performance metrics between different training runs (e.g., before and after adding more data)
  - Query params:
    - `model_type` (default: 'random_forest')
    - `baseline_tag` (default: 'baseline')
    - `comparison_tag` (required) - Tag of the run to compare against baseline
- `/update-weights` - Update statistical weights based on ML model feature importance

## Potential Future Data Sources:

Paid:

- Wyscout https://www.hudl.com/products/wyscout
- Opta Sports API

Free:

- FBref
- Understat
- Whoscored

## Data Collection Ethics and Legality

### FBref Data Scraping

SoccerScout uses data from FBref.com for player statistics. My data collection approach is designed to be ethical and respectful:

1. **Rate Limiting**: My scraper implements proper rate limiting with random delays between requests (3-6 seconds) to avoid overloading FBref's servers.

2. **Respectful Access**: I follow robots.txt guidelines and only access publicly available data that doesn't require authentication.

3. **Minimal Requests**: I cache results and only request pages when necessary, minimizing server load.

4. **Attribution**: I clearly credit FBref as the data source and do not claim ownership of their data.

5. **No Competitive Product**: I am not creating a competitive product to FBref, only using their data for analysis.

6. **User-Agent Declaration**: I use a proper User-Agent string to identify my scraper.

I follow best practices to ensure my activities remain ethical and sustainable. For production use at larger scale, purchasing data from official providers like Opta or Wyscout would be implemented.

## Resources:

- [Scraping FBref with Python](https://medium.com/@ricardoandreom/how-to-scrape-and-personalize-data-from-fbref-with-python-a-guide-to-unlocking-football-insights-7e623607afca)
- [Soccer API Scraping Guide](https://ctomashot.medium.com/how-i-scraped-an-api-for-my-soccer-scouting-app-c67df68da6ca)
- [Machine Learning Algorithms Overview](https://builtin.com/data-science/tour-top-10-algorithms-machine-learning-newbies)
- [Web Scraping Best Practices](https://www.scrapehero.com/how-to-prevent-getting-blacklisted-while-scraping/)
