""" we are trying to predict the number of medals and we will use  no of
medals and athelets"""

import pandas as pd
teams = pd.read_csv("teams.csv")

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
print(teams)

#to find the co-relation of medals with all the other coloumns
#we use numeric only because .corr() function does not automatically ignore the strings
print(teams.corr(numeric_only=True)["medals"])

import seaborn as sns
import matplotlib.pyplot as plt
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
#plt.show()

sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
#plt.show()

#to chech if our data is balanced
teams.plot.hist(y="medals")
#plt.show()

#find if there are rows with missing values
print(teams[teams.isnull().any(axis=1)])
#drops rows with missing values in our case countries which did not participate previously
teams = teams.dropna()
print(teams)

#split data
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

print(train.shape)
print(test.shape)

#training the model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

predictors = ["athletes", "prev_medals"]
target = "medals"

reg.fit(train[predictors], train["medals"])
LinearRegression()

predictions = reg.predict(test[predictors])
print(predictions)

#first we added the predictions as the rightmost coloumn in our test data to easily view
test["predictions"] = predictions
print(test)

test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

print(test)

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"], test["predictions"])
print("error = ", error)

#trying to check error with standard deviation
print(teams.describe()["medals"])

#checking predictions team by team
print(test[test["team"] == "USA"])
print(test[test["team"] == "IND"])

errors = (test["medals"] - test["predictions"]).abs()
print(errors)
error_by_team = errors.groupby(test["team"]).mean()
print(error_by_team)
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team / medals_by_team
print(error_ratio)

import numpy as np
error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]
print(error_ratio)

error_ratio.plot.hist()
plt.show()

#to improve the accuracy
#we can add more predictors
#or we can try differnt models
