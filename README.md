# Stock Market Prediction with Random Forest

This project demonstrates a machine learning approach to predicting stock market closing prices using a Random Forest model. It includes Python scripts for training the model with historical stock data and for making real-time predictions via an API.

## Project Structure
- `main.py`: Python script to train the Random Forest model using historical stock data.
- `stock-market-api.py`: Python script that fetches live stock data from an API and uses the trained model to make predictions.
- `TCS.csv`: A sample historical dataset used to train and evaluate the model.

## Prerequisites
- Python 3.7 or higher
- Required Python libraries (listed in `requirements.txt`)

## Setup Instructions
1. **Clone the Repository**  
   Clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/mittal-shreya/stock-market-prediction-model.git
   ```

2. **Navigate to Project Directory**  
   Move into the project folder:
   ```bash
   cd stock-market-prediction-model
   ```

3. **Create and Activate a Virtual Environment (Optional but Recommended)**  
   Create a virtual environment and activate it to isolate the project's dependencies:
   - Linux/Mac:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

4. **Install Required Dependencies**  
   Install all the required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Training the Model
- Run `main.py` to train the Random Forest model using the historical dataset (`TCS.csv`):
```bash
python main.py
```
- The trained model will be saved as `random_forest_stock_model.pkl` for future use.

### 2. Making Real-Time Predictions
- Execute `stock-market-api.py` to fetch live stock data and make predictions:
```bash
python stock-market-api.py
```

### 3. API Details
- Modify the API URL in `stock-market-api.py` to fetch the desired stock data. Ensure that your API key is configured correctly.

## Project Details
### Model Evaluation
The model's performance is measured using metrics like:
- **Root Mean Squared Error (RMSE)**: Measures the average error between actual and predicted prices.
- **Mean Absolute Error (MAE)**: Indicates the mean absolute difference between actual and predicted values.
- **R-squared Score (R²)**: Represents the proportion of variance in the dependent variable explained by the model.

### Features Used
The features include:
- **Previous Close Price:** The closing price on the previous day.
- **Previous Volume:** The trading volume of the previous day.
- **SMA (5-day):** 5-day Simple Moving Average.
- **EMA (10-day):** 10-day Exponential Moving Average.

### Future Improvements
Potential improvements to enhance this project could include:
- Experimenting with different machine learning models and comparing their performance.
- Adding more financial features and indicators to improve prediction accuracy.
- Developing a web application to interactively display stock predictions.

