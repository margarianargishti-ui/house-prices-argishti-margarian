# House Prices: Advanced Regression Techniques

**Author:** Argishti Margarian
**Group:** AI 2.4
**Competition:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## 1. Project overview

The goal of this project is to predict the sale price of houses using 79 features that describe each house, such as its size, quality, age, location, etc.

The training data contains 1460 houses with known prices, and the test data contains 1459 houses whose prices we predict. Evaluation of the predictions with the RMSE between the logarithm of the predicted price and the logarithm of the actual price. Errors on cheap and expensive houses are weighted equally.

## 2. Project structure

- **data/raw/** holds the four original train.csv, test.csv, sample_submission.csv, data_description.txt files.
- **data/processed/** holds train_processed.csv and test_processed.csv, the cleaned data created by the 01_EDA.ipynb notebook.
- **notebooks/01_EDA.ipynb** Exploration, cleaning, new features and encoding.
- **notebooks/02_Models.ipynb** Models, tuning, comparison and error analysis.
- **results/figures/** holds the saved plots in png format.
- **results/model_comparison.csv** lists all models with their cv scores.
- **submissions/submission_blend.csv** is the file uploaded to Kaggle.
- **submissions/kaggle_score.png** is a screenshot of the Kaggle score.
- **requirements.txt** lists the Python libraries needed for the project.

## 3. How to run

1. Download the four data files from the Data tab of the Kaggle competition and place them in the data/raw folder.
2. Install the required libraries listed in requirements.txt (pandas, NumPy, matplotlib, seaborn, scikit-learn and Jupyter).
3. Open the notebooks folder and run 01_EDA.ipynb. It creates the processed data.
4. Then run 02_Models.ipynb. It trains and compares the models and saves the comparison table.

## 4. Method

### EDA (01_EDA.ipynb)

**Target -** SalePrice is strongly right-skewed. After a log transformation the skew drops, so we model the log of the price.

**Outliers.** Two training houses with unrealistic prices were removed, meaning they were dropped from the training data.

**Missing values.** A missing value usually means the house does not have that feature. These were filled with "None" for text columns and 0 for numeric columns. Missing LotFrontage values were filled with the median of the same neighborhood.

**New features.** Nine new columns were created: total square footage, total bathrooms, house age, years since remodeling, total porch area, quality multiplied by living area, and flags for having a pool, a garage or a second floor. The new features correlate more strongly with price than the original columns. Quality times area and total square footage are the most important features of the boosting model, and the cross-validated error improved.

**Encoding.** Quality ratings were converted to numbers that keep their order. All remaining text columns were one-hot encoded on the training and test data together, so both have exactly the same 267 columns.

### Modeling (02_Models.ipynb)

All models were evaluated with K-fold CV (K=5), using the same folds for every model and RMSE on log price as the score. The models were a baseline that always predicts the average price, Ridge regression, Lasso regression, Random Forest and Gradient Boosting. The linear models scale the data inside a pipeline to avoid data leakage. Finally, the tuned Lasso and Gradient Boosting models were combined by averaging their predictions, as they show the most optimal result.

## 5. Results

CV RMSE on log SalePrice:

- Blend of Lasso and Gradient Boosting: 0.1090
- Lasso, tuned with alpha 0.0005: 0.1126
- Ridge, tuned with alpha 15: 0.1141
- Gradient Boosting, tuned with depth 3 and learning rate 0.03: 0.1145
- Random Forest: 0.1306
- Baseline: 0.3993

Lasso + Gradient Boosting droped our baseline error the most (73%).

### Key findings

- Overall quality (corr 0.80) and living area (corr 0.73) are the strongest drivers of price, and the median price per neighborhood ranges from 88000 to 315000.
- Random Forest is the weakest model because it cannot predict beyond the range of prices it has seen. Linear models were best.
- Combining a linear model with a tree model gives the best result because the two make different kinds of mistakes.
- The largest prediction errors are on cheap houses sold under unrealistic conditions. 

### Kaggle score

**Public leaderboard RMSE:** 0.12462

## 7. Tools

- Python
- Numpy (np)
- Pandas (pd)
- Matplotlib (plt)
- Seaborn (sns)
- Scikit-Learn (sk-learn)
- Jupyter Notebook

## 8. Sources

- Kaggle - *House Prices – Advanced Regression Techniques*, competition page and data_description.txt
- Claude Code - EDA plottings, creating this README and explination of the machine learning methods used.
- scikit-learn documentation: https://scikit-learn.org/stable/
- pandas documentation: https://pandas.pydata.org/docs/
- seaborn documentation: https://seaborn.pydata.org/
- numpy documentation: https://numpy.org/doc/stable/
- matplotlib documentation: https://matplotlib.org/stable/index.html