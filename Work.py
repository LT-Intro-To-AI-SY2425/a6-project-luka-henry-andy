import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("insurance.csv")

# Encode categorical columns: sex, smoker, region
data_encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# Define features (x) and target (y) - 'charges' is the target variable
x = data_encoded[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]].values
y = data_encoded["charges"].values  # Use 'charges' as the target variable

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression().fit(xtrain, ytrain)

# Get the coefficients and intercept
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y), 2)

# Print the model's linear equation
print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {coef[4]}x5 + {coef[5]}x6 + {coef[6]}x7 + {coef[7]}x8 + {intercept}")
print("R Squared value:", r_squared)

# Predict on the test set
predict = model.predict(xtest)
predict = np.around(predict, 2)
print("Predicted charges:", predict)

# Create subplots for visualizing the relationship between each feature and charges
fig, graph = plt.subplots(3, 2, figsize=(12, 10))

# Features and their corresponding indices in x
features = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest']

# Plot each feature vs charges
for i, ax in enumerate(graph.flatten()):
    feature = features[i]
    x_feature = xtrain[:, i]  # Only use the training set for plotting
    
    # Scatter plot of the feature vs charges
    ax.scatter(x_feature, ytrain, color='blue', label=f'{feature} vs Charges')
    
    # Set the x and y labels
    ax.set_xlabel(feature)
    ax.set_ylabel("Charges")
    
    # Create the best-fit line for each feature using the full model
    # We create a version of the data with only the current feature changing while the rest are constant
    x_temp = np.copy(xtrain)
    for j in range(x_temp.shape[1]):
        if j != i:
            x_temp[:, j] = np.mean(xtrain[:, j])  # Set all other features to their mean values
    
    # Predict the charges for the modified x_temp
    y_fit = model.predict(x_temp)
    
    # Plot the best-fit line
    ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Charges)')
    ax.legend()

# Display the plot with tight layout
plt.tight_layout()
plt.show()
