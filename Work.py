import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("insurance.csv")
x = data[["age", "sex", "bmi", "children", "smoker", "region"]].values
y = data["price"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
#########################################

model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y), 2)

print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {coef[4]}x5 + {coef[5]}x6 + {intercept}")
print("R Squared value:", r_squared)

predict = model.predict(xtest)
predict = np.around(predict, 2)
print("Predicted prices:", predict)

# Use a 2D array as input for model.predict
# specific_input = np.array([[150000, 50]])
# predict = model.predict(specific_input)
# predict = np.around(predict, 2)

print("Predicted price for 10 miles and 89,000 years old car:", predict)


# Create subplots: 3 rows and 2 columns for all 6 features
fig, graph = plt.subplots(3, 2, figsize=(12, 10))

# Set up each subplot for different features
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
x_data = [x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]]

for i, ax in enumerate(graph.flatten()):
    feature = features[i]
    x_feature = x_data[i]
    
    # Scatter plot
    ax.scatter(x_feature, y, color='blue')
    ax.set_xlabel(feature)
    ax.set_ylabel("Price")
    
    # Create the best-fit line
    y_fit = coef[i] * x_feature + intercept
    ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Price)')
    ax.legend()

# Display the plot with tight layout
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