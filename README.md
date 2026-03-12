# Salary Prediction Model

This project implements a simple linear regression model in Python to predict salaries based on years of experience. 

## Overview

The `newmodel.py` script trains a machine learning model using a dataset (`Salary_dataset.csv`) that maps years of experience to salary. It visualizes the relationship between the two variables and evaluates the model's accuracy using the R-squared score.

## Dependencies

Make sure you have the following Python libraries installed before running the script:

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install scikit-learn pandas numpy matplotlib
```

## How It Works

1. **Loads Data**: Reads the `Salary_dataset.csv` file containing `YearsExperience` and `Salary` columns.
2. **Preprocesses Data**: Converts the data into arrays and reshapes them for the model to use.
3. **Trains Model**: Splits the dataset into training and testing sets (75% training, 25% testing by default) and fits a `LinearRegression` model.
4. **Evaluates Performance**: Predicts salary values against the test set and outputs the R-squared 
(0.9756) score to measure accuracy.
5. **Generates Visualization**: Creates a scatter plot charting the data points alongside the linear regression line.

## Usage

To train the model and view the results, run the script from your terminal:

```bash
python newmodel.py
```

The script will output the R-squared score to the console and present a Matplotlib window illustrating the regression analysis.
