import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# if all else fails run pip install scikit-learn

data = pd.read_csv("insurance.csv")

data_encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

x = data_encoded[["age", "bmi", "smoker_yes"]].values  # Use age, bmi, smoker as features
y = data_encoded["charges"].values  # Use 'charges' as the target variable

x = np.column_stack((data_encoded["age"], data_encoded["bmi"], data_encoded["smoker_yes"]))
features = ['age', 'bmi', 'smoker']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(xtrain, ytrain), 2)

print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("R Squared value:", r_squared)

fig, graph = plt.subplots(1, 3, figsize=(12, 5))  # Adjusted to 1x3 for 3 features

x_data = [xtrain[:, 0], xtrain[:, 1], xtrain[:, 2]]  # 'age', 'bmi', 'smoker_yes'

for i, ax in enumerate(graph.flatten()):
    feature = features[i]
    x_feature = x_data[i]  
    
    ax.scatter(x_feature, ytrain, color='blue', label=f'{feature} vs Charges')
    
    ax.set_xlabel(feature)
    ax.set_ylabel("Charges")
    
    x_temp = np.copy(xtrain)
    for j in range(x_temp.shape[1]):
        if j != i:
            x_temp[:, j] = np.mean(xtrain[:, j])  # Set all other features to their mean values
    
    y_fit = model.predict(x_temp)
    
    ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Charges)')
    ax.legend()

plt.tight_layout()
plt.show()
 
#QUESTIONNAIRE

userAge = input("Please enter your age")
userAge = int(userAge)

userBmi = input("what is your BMI")
userBmi = int(userBmi)

userSmoker = input("Please enter 'yes' if you smoke and 'no' if you do not smoke.").lower()
if userSmoker == "yes":
    userSmoker = 1
elif userSmoker == "no":
    userSmoker = 0
else:
    print("Sorry, I didn't understand that. Please enter 'yes' or 'no'.")
    userSmoker = input("Please enter 'yes' or 'no': ").lower()



#Predictions
person_data = np.array([[userAge, userBmi, userSmoker]])  # 50 years old, BMI 30, smoker (smoker = 1)
predicted_charge = model.predict(person_data)
predicted_charge_rounded = np.round(np.maximum(0, predicted_charge), 2)
print(f"The predicted charge for the person is: ${predicted_charge_rounded[0]}")

#-------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Load the data
# data = pd.read_csv("insurance.csv")

# # Encode categorical columns: sex, smoker, region
# data_encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# # Define features (x) and target (y) - 'charges' is the target variable
# x = data_encoded[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]].values
# y = data_encoded["charges"].values  # Use 'charges' as the target variable

# # Split the data into training and test sets
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# # Train the linear regression model
# model = LinearRegression().fit(xtrain, ytrain)

# # Get the coefficients and intercept
# coef = np.around(model.coef_, 2)
# intercept = round(float(model.intercept_), 2)
# r_squared = round(model.score(xtrain, ytrain), 2)

# # Print the model's linear equation
# print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {coef[4]}x5 + {coef[5]}x6 + {coef[6]}x7 + {coef[7]}x8 + {intercept}")
# print("R Squared value:", r_squared)

# # Predict on the test set
# predict = model.predict(xtest)
# predict = np.around(predict, 2)
# print("Predicted charges:", predict)

# # Create subplots for visualizing the relationship between each feature and charges
# fig, graph = plt.subplots(3, 2, figsize=(12, 10))

# # Features and their corresponding indices in x
# features = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest']
# x_data = [xtrain[:, 0], xtrain[:, 1], xtrain[:, 2], xtrain[:, 3], xtrain[:, 4], xtrain[:, 5]]

# # Plot each feature vs charges
# for i, ax in enumerate(graph.flatten()):
#     feature = features[i]
#     x_feature = x_data[i]  # Use training data for each feature
    
#     # Scatter plot of the feature vs charges
#     ax.scatter(x_feature, ytrain, color='blue', label=f'{feature} vs Charges')
    
#     # Set the x and y labels
#     ax.set_xlabel(feature)
#     ax.set_ylabel("Charges")
    
#     # Create the best-fit line for each feature using the full model
#     # We create a version of the data with only the current feature changing while the rest are constant
#     x_temp = np.copy(xtrain)
#     for j in range(x_temp.shape[1]):
#         if j != i:
#             x_temp[:, j] = np.mean(xtrain[:, j])  # Set all other features to their mean values
    
#     # Predict the charges for the modified x_temp
#     y_fit = model.predict(x_temp)
    
#     # Plot the best-fit line (single line for the feature)
#     ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Charges)')
#     ax.legend()

# # Display the plot with tight layout
# plt.tight_layout()
# plt.show()

#------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Load the data
# data = pd.read_csv("insurance.csv")

# # Encode categorical columns: sex, smoker, region
# data_encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# # Define features (x) and target (y) - 'charges' is the target variable
# x = data_encoded[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]].values
# y = data_encoded["charges"].values  # Use 'charges' as the target variable

# # Split the data into training and test sets
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# # Train the linear regression model
# model = LinearRegression().fit(xtrain, ytrain)

# # Get the coefficients and intercept
# coef = np.around(model.coef_, 2)
# intercept = round(float(model.intercept_), 2)
# r_squared = round(model.score(x, y), 2)

# # Print the model's linear equation
# print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {coef[3]}x4 + {coef[4]}x5 + {coef[5]}x6 + {coef[6]}x7 + {coef[7]}x8 + {intercept}")
# print("R Squared value:", r_squared)

# # Predict on the test set
# predict = model.predict(xtest)
# predict = np.around(predict, 2)
# print("Predicted charges:", predict)

# # Create subplots for visualizing the relationship between each feature and charges
# fig, graph = plt.subplots(3, 2, figsize=(12, 10))

# # Features and their corresponding indices in x
# features = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest']

# # Plot each feature vs charges
# for i, ax in enumerate(graph.flatten()):
#     feature = features[i]
#     x_feature = xtrain[:, i]  # Only use the training set for plotting
    
#     # Scatter plot of the feature vs charges
#     ax.scatter(x_feature, ytrain, color='blue', label=f'{feature} vs Charges')
    
#     # Set the x and y labels
#     ax.set_xlabel(feature)
#     ax.set_ylabel("Charges")
    
#     # Create the best-fit line for each feature using the full model
#     # We create a version of the data with only the current feature changing while the rest are constant
#     x_temp = np.copy(xtrain)
#     for j in range(x_temp.shape[1]):
#         if j != i:
#             x_temp[:, j] = np.mean(xtrain[:, j])  # Set all other features to their mean values
    
#     # Predict the charges for the modified x_temp
#     y_fit = model.predict(x_temp)
    
#     # Plot the best-fit line
#     ax.plot(x_feature, y_fit, color='red', label=f'Best Fit Line ({feature} vs Charges)')
#     ax.legend()

# # Display the plot with tight layout
# plt.tight_layout()
# plt.show()
