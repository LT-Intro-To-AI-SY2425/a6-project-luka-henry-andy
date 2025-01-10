import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("insurance.csv")

# Encode categorical columns
data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# Prepare the features and target
x = data[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northeast", "region_northwest", "region_southeast", "region_southwest"]].values
y = data["price"].values

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

# Train the model
model = LinearRegression().fit(xtrain, ytrain)

# Get coefficients and intercept
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y), 2)

# Print the model equation
print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {coef[4]}x5 + {coef[5]}x6 + {coef[6]}x7 + {coef[7]}x8 + {coef[8]}x9 + {intercept}")
print("R Squared value:", r_squared)

# Predict on the test set
predict = model.predict(xtest)
predict = np.around(predict, 2)
print("Predicted prices:", predict)

# Plot the results
fig, graph = plt.subplots(3, 2, figsize=(12, 10))

features = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northeast']
x_data = [x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]]

for i, ax in enumerate(graph.flatten()):
    feature = features[i]
    x_feature = x_data[i]
    
    ax.scatter(x_feature, y, color='blue')
    ax.set_xlabel(feature)
    ax.set_ylabel("Price")
    
    y_fit = coef[i] * x_feature + intercept
    ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Price)')
    ax.legend()

plt.tight_layout()
plt.show()







# x_1 = x[:, 0]  
# x_2 = x[:, 1]  
# x_3 = x[:, 2]
# x_4 = x[:, 3]
# x_5 = x[:, 4]
# x_6 = x[:, 5]

# fig, graph = plt.subplots(2, 1, figsize=(8, 6))

# graph[0].scatter(x_1, y, color='blue')
# graph[0].set_xlabel("Total Miles")
# graph[0].set_ylabel("Price")

# y_fit_1 = coef[0] * x_1 + intercept
# graph[0].plot(x_1, y_fit_1, color='red', label='Best Fit Line (Miles vs Price)')
# graph[0].legend()

# graph[1].scatter(x_2, y, color='green')
# graph[1].set_ylabel("Price")
# graph[1].set_xlabel("Car Age")

# y_fit_2 = coef[1] * x_2 + intercept
# graph[1].plot(x_2, y_fit_2, color='red', label='Best Fit Line (Age vs Price)')
# graph[1].legend()

# corr_miles = np.corrcoef(x_1, y)[0, 1]
# corr_age = np.corrcoef(x_2, y)[0, 1]

# print(f"Correlation between Total Miles and Car Price: {round(corr_miles, 3)}")
# print(f"Correlation between Age and Car Price: {round(corr_age, 3)}")

# plt.tight_layout()
# plt.show()