# Super Bowl Predictor

This project is a machine learning-based predictor for Super Bowl scores, specifically designed to predict the scores for the Philadelphia Eagles and the Kansas City Chiefs. The prediction is based on historical and seasonal NFL team statistics.

## Features

- **Data Collection**: Retrieves NFL team statistics and game details using a REST API.
- **Data Processing**: Cleans and normalizes data using MinMaxScaler and OneHotEncoder.
- **Model Training**: Utilizes Linear Regression to train models on historical data.
- **Score Prediction**: Predicts scores for specified teams using trained models.

## Requirements

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `python-dotenv`, `http.client`, `json`, `os`

## Setup

1. **Clone the Repository**: 
  ```bash
  git clone https://github.com/Awnder/superbowl_predictions.git
  cd superbowl_predictions
  ```

2. **Environment Variables**:
  ```bash
  RAPID_NFL_API_KEY=<your-api-key>
  RAPID_NFL_API_HOST=<your-api-host>
  ```

## Usage
1. Run the script with `python superbowl_predictor.py`

2) Output: The script will print predicted scores for the Philadelphia Eagles and the Kansas City Chiefs for the upcoming Super Bowl.

3) You can change the prediction teams by editing the line `PREDICTION_NAMES` in the `main()` function.