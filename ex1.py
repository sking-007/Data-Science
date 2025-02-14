import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import matplotlib as plt

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Reading Data from the file
data = pd.read_csv('cardio_train.csv', sep=';')

# Create a Logistic Regression Model
scaler = StandardScaler()
model = LogisticRegression()

# Printing dimension from the dataframe
print("data.shape: ", data.shape)

# Printing Columns from dataframe
for col in data.columns:
    print(col)

# Selecting values of the columns age into an array
age = np.array([data['age']])
print("age;", age[0:10, ])

# Computing the year of the age values
data['age'] = np.round((data['age'] / 365.24), 0)
print("age", data['age'][0:10])
# Computing the mean of the age values
print("mean age\n", np.mean(data['age']))

# Compute maximum and minimum of the age values
print("min age\n", np.min(data['age']))
print("max age\n", np.max(data['age']))

y = data['cardio']
X = data.drop(['cardio'], axis=1).copy()

X_scaled = scaler.fit_transform(X)

# Splitting data into training and test data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, random_state=9, shuffle=False)
print("np.std(X_train)\n", np.std(X_train))
print('X_train:')
print(X_train)
print(X_test)
print("y_train:")
print(y_train)
print(y_test)

# Scale data using standard scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train the model
model.fit(X_train_scaled, y_train)

# Predict the output
y_pred = model.predict(X_test_scaled)
log_model = sm.Logit(y_train, sm.add_constant(X_train))
log_result = log_model.fit()
print(log_result.summary2())
print("prediction")
conf_matrix = confusion_matrix(y_test, y_pred)
print("conf_matrix\n", conf_matrix)
print("Exponents of coefficients\n", np.exp(log_result.params).sort_values(ascending=False))
coefs = log_result.params.drop(labels=['const', 'gender'])
stdv = np.std(X_train).drop(labels='gender')
print("Impact: coefs *stdv\n", abs(coefs * stdv).sort_values(ascending=False))
print("\n")
print("Exponents of coefficients\n", np.exp(log_result.params).sort_values(ascending=False))
print("\n")
log_model = sm.Logit(y_train, sm.add_constant(X_train))
log_result = log_model.fit()
print(log_result.summary2())
# Standard Derivation
print("\n")
print("np.std(X_train)\n", np.std(X_train))
print("\n")
print("np.std(X_train)\n"
      , np.std(X_train))
# Logistic Regression Model
print("\n")

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)
print("conf_matrix\n", conf_matrix)

importances = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance':
    model.coef_[0]})

importances = importances.sort_values(by='Importance', ascending=False)
print("\nimportances coefficients Logistic Regression-based model\n", importances)
coefs = log_result.params.drop(labels=['const', 'gender'])
stdv = np.std(X_train).drop(labels='gender')
print("Impact: coefs *stdv\n", abs(coefs * stdv).sort_values(ascending=False))

# Print the importance of the features
importances = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance':
    model.coef_[0]})
importances = importances.sort_values(by='Importance', ascending=False)
print("\nimportances coefficients Logistic Regression-based model\n", importances)

y_pred = log_result.predict(sm.add_constant(X_test))
print("y_pred.shape:", y_pred.shape)
print("\ny_pred\n", y_pred)

# Print all features and their values of one proband
print("y_pred[63000]\n", y_pred[63000])

importances = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance':
    model.coef_[0]})
# 1. coefficients
importances = importances.sort_values(by='Importance', ascending=False)
print("\nimportances coefficients Logistic Regression-based model\n", importances)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances Logistic Regression-based model', size=20)
plt.xticks(rotation='vertical')
plt.show()