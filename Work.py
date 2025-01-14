import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# if all else fails run pip install scikit-learn

data = pd.read_csv("insurance.csv")

data_encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

x = data_encoded[["age", "bmi", "smoker_yes"]].values  
y = data_encoded["charges"].values  

x = np.column_stack((data_encoded["age"], data_encoded["bmi"], data_encoded["smoker_yes"]))
features = ['age', 'bmi', 'smoker']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(xtrain, ytrain)

coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(xtrain, ytrain), 2)

print(f"Model's Linear Equation: y = {coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("R Squared value:", r_squared)

fig, graph = plt.subplots(1, 3, figsize=(12, 5))  

x_data = [xtrain[:, 0], xtrain[:, 1], xtrain[:, 2]]  

for i, ax in enumerate(graph.flatten()):
    feature = features[i]
    x_feature = x_data[i]  
    
    ax.scatter(x_feature, ytrain, color='blue', label=f'{feature} vs Charges')
    
    ax.set_xlabel(feature)
    ax.set_ylabel("Charges")
    
    x_temp = np.copy(xtrain)
    for j in range(x_temp.shape[1]):
        if j != i:
            x_temp[:, j] = np.mean(xtrain[:, j])
    
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
person_data = np.array([[userAge, userBmi, userSmoker]])  
predicted_charge = model.predict(person_data)
predicted_charge_rounded = np.round(np.maximum(0, predicted_charge), 2)
print(f"The predicted charge for the person is: ${predicted_charge_rounded[0]}")

