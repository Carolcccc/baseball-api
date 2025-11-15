brfss_cancer_data.csv: The clean dataset used in Tableau.

# Cancer Prediction Project

This project uses BRFSS survey data to predict cancer risk using logistic regression. The workflow includes data cleaning, feature selection, model training, and saving results for further analysis or visualization.

## Project Structure

- `cancer_prediction.py` — Main script for data processing and model training
- `LLCP2024.XPT` — Main data file (SAS XPT format)
- `brfss_cancer_data.csv` — Cleaned dataset for analysis/visualization
- `cancer_risk_model.pkl` — Trained machine learning model
- `model_features.pkl` — List of model features
- `column_names.txt` — All column names in the dataset

## Requirements

- Python 3.11+
- pandas
- scikit-learn

Install requirements with:

```
pip install pandas scikit-learn
```

## Usage

1. Place the `LLCP2024.XPT` file in the project directory.
2. Run the main script:

```
python cancer_prediction.py
```

3. The script will:
	- Load and clean the data
	- Print the number of records per state
	- Train a logistic regression model to predict cancer diagnosis
	- Save the cleaned data and model files

## Feature Columns Used

- `CHCSCNC1` — Ever told you had skin cancer
- `_SMOKER3` — Smoking status
- `_AGE_G` — Age group
- `_SEX` — Sex
- `_BMI5` — BMI
- `_STATE` — State code

## Output

- Cleaned data: `brfss_cancer_data.csv`
- Model: `cancer_risk_model.pkl`
- Model features: `model_features.pkl`
- Column names: `column_names.txt`

## Notes

- The script prints a classification report after model training.
- You can adjust the feature columns in `cancer_prediction.py` as needed.
- For visualization, use the CSV output in Tableau or similar tools.

---

For questions or improvements, please open an issue or contact the author.