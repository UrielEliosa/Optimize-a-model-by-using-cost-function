import pandas
import wget
from datetime import datetime
import numpy as np
from numpy import doc
from microsoft_custom_linear_regressor import MicrosoftCustomLinearRegressor 
import graphing

# decision = input("Desea Descargar los datos?: ")

# if decision == "si":
#     wget.download("https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py")
#     wget.download("https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/microsoft_custom_linear_regressor.py")
#     wget.download("https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv")

dataset = pandas.read_csv('seattleWeather_1948-2017.csv', parse_dates=['date'])

dataset["year"] = [(d.year + d.timetuple().tm_yday / 365.25) for d in dataset.date]


desired_dates = [
    datetime(1950,2,1),
    datetime(1960,2,1),
    datetime(1970,2,1),
    datetime(1970,2,1),
    datetime(1980,2,1),
    datetime(1990,2,1),
    datetime(2000,2,1),
    datetime(2010,2,1),
    datetime(2017,2,1),
]

dataset = dataset[dataset.date.isin(desired_dates)].copy()

print(dataset)

def sum_of_square_differences(estimate,actual):
    return np.sum((estimate - actual)**2)

def sum_of_absolute_differences(estimate,actual):
    return np.sum(np.abs(estimate - actual))

# model_estimate = np.array([1,1])
# actual_label = np.array([1,3])

# print("SSD:", sum_of_square_differences(model_estimate,actual_label))
# print("SAD:", sum_of_absolute_differences(model_estimate,actual_label))

model = MicrosoftCustomLinearRegressor().fit(X = dataset.year,
                                             y = dataset.min_temperature,
                                             cost_function = sum_of_square_differences)

graphing.scatter_2D(dataset, 
                    label_x="year",
                    label_y="min_temperature",
                    trendline= model.predict)
