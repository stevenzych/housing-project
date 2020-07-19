# *King County Housing Prices: Linear Regression For First-Time Buyers*
**Steven Zych - July 2020**

# Introduction

This project looks at housing prices in King County, Washington--which includes Seattle, Bellevue, Renton, Tacoma, and a handful of smaller cities and countryside. Housing data was sourced from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction), manipulated and cleaned in Python as a Pandas dataframe, and ultimately modelled in an iterative approach using StatsModels and Scikit-learn.

More specifically, this project looks at best choices for first-time buyers in the Seattle area, and aims to answer the question:
> "What factors bring down the price of a house the most, with least impact to overall quality?"

Obviously, this is a fairly subjective question, but categories will be posited later that aim to answer which variables count as having "least impact" on the quality of a home.

All in all, the following packages were used:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- StatsModels
- Scikit-Learn
- Warnings
- Datetime

# EDA and Cleaning 

## Column Names

The first (and perhaps most important) part of cleaning this data was sorting through the columns, identifying their meaning, and determining their usefulness to the project. The full list is as follows:
`['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']`
Columns with `15` at the end refer to the values for the nearest 15 neighbors, waterfront is a yes-no column of whether or not you're on it, and view *claims* to represent whether or not the home has been viewed, yet the values range from 0-4. (Almost every value was 0, and this column was later dropped.)

## Data Types, Duplicates, And Missing Values

The `date` column was changed from strings to datetimes, `sqft_basement` was purged of an erroneous `?` value in many of its rows, and `date` was further manipulated into being represented as `day_of_year` (1-365) so that the model could use it better for predictions (as opposed to interpteting `02-15-2011`, for example).

There were no fully-duplicated rows in the data that needed to be expunged, but there were 117 rows with duplicate ID's--likely homes that were resold. These rows were dropped from the data as they represented only 0.01% of it.

The columns `waterfront` and `yr_renovated` contained thousands of missing values each, both sets of which were replaced with the most likely outcome--0. This indicated "Not on the waterfront" and "Never been renovated" respectively.

With the data cleaned, a baseline model could be built.

# Baseline Model

The first model takes all the data as-is, and uses every non-price column to predict the `price` column. No special modifications made, just plain and simple. The model's R-squared value was **0.692.** For the uninitiated, an R-squared value of **1** would mean that the linear regression model matches onto the predicted value (`price` here) *perfectly.* This is pretty unheard of. For a baseline model however, the R-squared of 0.692 is *not bad,* but we can do better. The following models aim to do just that.

# Model 2: Mean Normalization of Continuous Variables

What did I do to the model? (Which model was this based on)
How did it affect the model? (R^2, RMSE, Was this model kept?)
A picture of it.

# Model 3: Categorical Variables As Dummies

**Example of how to put image in**
![Scatter Plot Of Runtime And ROI](/readme_images/runtime_scatter.PNG)

# Model 4: Selective Predictors

# Model 5: Only P-Values Under 0.05

# Model 6: Log Transformations On Continuous Variables

# Final Model

# Recommendations

To maximize your money buying a house in your Seattle, I urge you to do the following:
1.
1.
1.
1.

# Future Research

When next I aid first-time buyers in house-hunting in the greater Seattle area, I'd like to devote more time to feature-engineering and/or data-scraping, as well as zipcodes as a predictor of price.

In this research, the only spatial data points used to predict price were latitude and longitude which--while salient--only provide broad strokes of information. Predicting instead by zipcodes and aggregated zipcodes (such that they represent whole neighborhoods, suburbs, and discrete cities) would potentially be a greater predictor of price since similar areas would be grouped with similar areas. In other words, the reputation of "living in Renton" vs. "living in Bellevue," whatever that may mean, would be more present in the data when these spatial groups are recreated in it.

Two final spatial considerations would be proximity to cultural institutions and main streets. Does living next to Pike Place Market drive the price of your house up? What about the Olympic Sculpture park? How about if you live on Pine Street near all the cool restaurants? Intuition says "Yes" to all these, but the data has yet to speak.

# Conclusion

Thank you for lending me your time and consideration, and--more importantly--your trust. It is my sincere hope that the predictions and models set forth in this paper are beneficial to all the first-time buyers reading. Happy house hunting!