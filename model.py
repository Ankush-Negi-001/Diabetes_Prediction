# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from warnings import filterwarnings
filterwarnings(action='ignore')

# Load the dataset
dataset = pd.read_csv("Diabetes_dataset.csv")

# Replace zero values with NaN
dataset[['Glucose', 'BloodPressure', 'Insulin', 'BMI']] = dataset[
    ['Glucose', 'BloodPressure', 'Insulin', 'BMI']].replace(0, np.NaN)

# Replace NaN with mean values
dataset['Glucose'].fillna(dataset['Glucose'].mean(), inplace=True)
dataset['BloodPressure'].fillna(dataset['BloodPressure'].mean(), inplace=True)
# dataset['SkinThickness'].fillna(dataset['SkinThickness'].mean(), inplace=True)
dataset['Insulin'].fillna(dataset['Insulin'].mean(), inplace=True)
dataset['BMI'].fillna(dataset['BMI'].mean(), inplace=True)


# dataset.head()


Outcome = dataset['Outcome']
data = dataset[['Glucose', 'BloodPressure', 'Insulin', 'BMI']]



z_scores = np.abs((data - data.mean()) / data.std())
threshold = 3
outlier_indices = np.where(z_scores > threshold)


print(f"Number of outliers detected: {len(outlier_indices[0])}")
clear_data= data.drop(outlier_indices[0])
print(f"Number of data points after removing outliers: {len(clear_data)}")

dataset = pd.concat([clear_data, Outcome], axis=1)
 

dataset[['Glucose', 'BloodPressure', 'Insulin', 'BMI']] = dataset[
    ['Glucose', 'BloodPressure', 'Insulin', 'BMI']].replace(0, np.NaN)


# Replace NaN with mean values
dataset['Glucose'].fillna(dataset['Glucose'].mean(), inplace=True)
dataset['BloodPressure'].fillna(dataset['BloodPressure'].mean(), inplace=True)
dataset['Insulin'].fillna(dataset['Insulin'].mean(), inplace=True)
dataset['BMI'].fillna(dataset['BMI'].mean(), inplace=True)

# Split the dataset into input features (X) and target variable (y)
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
# dataset.head()


# Perform feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the models
models = [
    LogisticRegression(random_state=42, solver='liblinear')
]
# Evaluate models using cross-validation
def evaluate_models(models):

    kfold = StratifiedKFold(n_splits = 4)
    result = []
    
    for model in models :
        cv_results = cross_val_score(estimator = model, X = X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4)
        result.append(cv_results.mean())

    cv_means = []
    for cv_result in result:
        cv_means.append(cv_result.mean())
        
    result_df = pd.DataFrame({
        "CrossValMeans":cv_means,
        'Models': ['Logistic Regression'],#, 'SVC', 'Decision Tree', 'Random Forest'],
        'Accuracy': result
    })
    bar = sns.barplot(x = "CrossValMeans", y = "Models", data = result_df, orient = "h")
    bar.set_xlabel("Mean Accuracy")
    bar.set_title("Cross validation scores")
    return result_df

# Evaluate models
model_results = evaluate_models(models)
print(model_results)


# Hyperparameter tuning
def tune_parameters(model, params, cv,  num):
    if(num == 1):
       grid_search = GridSearchCV(estimator = model, param_grid = params , cv = cv, scoring = 'accuracy', error_score = 'raise')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score

# Fit the logistic regression model
models[0].fit(X_train, y_train)

# Evaluate models
model_results = evaluate_models(models)
print(model_results)

# Hyperparameter tuning for Logistic Regression
solvers = ['newton-cg', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, penalty=penalty, C=c_values)
cv = StratifiedKFold(n_splits=50, random_state=1, shuffle=True)
lr_best_params, lr_best_score = tune_parameters(models[0], grid, cv, 1)
print("Logistic Regression - Best Params:", lr_best_params)
print("Logistic Regression - Best Score:", lr_best_score)

# Fit the logistic regression model with best parameters
models[0].set_params(**lr_best_params)
models[0].fit(X_train, y_train)


# scaler = MinMaxScaler()
# X_test_scaled = scaler.fit_transform(X_test)

# Save the model
pickle.dump(models[0], open("model.pkl", "wb"))

# Load the model
loaded_model = pickle.load(open("model.pkl", "rb"))
