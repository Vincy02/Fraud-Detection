import inflection
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset_name = "Fraudulent_E-Commerce_Transaction_Data.csv"
data = pd.read_csv("../Data/"+dataset_name)

# Rimozione della feature non importante
data.drop(['IP Address'], axis=1, inplace=True)

# Rinominazione delle colonne
cols_old = data.columns.tolist()
snakecase = lambda x: inflection.parameterize(x, separator='_')
cols_new = list(map(snakecase, cols_old))
data.columns = cols_new

# Analisi dei dati a disposizione
from datasetAnalysis import analizeData, analizeTreshold, analizeAmountSuspicious, analizeLocations
analizeLocations(data.copy())
analizeAmountSuspicious(data.copy())
analizeTreshold(data.copy())
analizeData(data.copy())

# Creazione della KB in prolog
from prolog import create_kb, define_rules, consult_kb
create_kb(data.copy())
define_rules()
consult_kb(data.copy())

# Importazioni nuovo dataset
data = pd.read_csv("../Data/new_data.csv")

# Rimozione della feature non importanti (v2)
data.drop(['transaction_id', 'customer_id',
           'shipping_address', 'billing_address',
           'transaction_date', 'customer_location'], axis=1, inplace=True)

# Definizione delle colonne numeriche e categoriche
numeric_columns = ['transaction_amount',
                   'quantity',
                   'customer_age',
                   'account_age_days',
                   'transaction_hour',
                   'is_suspicious_transaction',
                   'high_risk_category_and_amount',
                   'suspicious_customer_location']

categorical_columns = ['payment_method',
                       'product_category',
                       'device_used']

X = data[categorical_columns + numeric_columns].copy()
y = data['is_fraudulent'].copy()

# Apprendimento NON supervisionato
_X = X.copy()
scaler = StandardScaler()
_X[numeric_columns] = scaler.fit_transform(_X[numeric_columns])

_X = pd.get_dummies(_X, columns=categorical_columns)
_X.columns = _X.columns.str.replace(' ', '_', regex=False)

from unSupervisedLearning import clusterDataAndVisualize
clusterDataAndVisualize(_X)

# Apprendimento supervisionato - preparazione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione valori numerici
scaler = StandardScaler()
scaler.fit(X_train[numeric_columns])
X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# One-hot encoder su dati categorici
X_train = pd.get_dummies(X_train, columns=categorical_columns)
X_test = pd.get_dummies(X_test, columns=categorical_columns)
X_train.columns = X_train.columns.str.replace(' ', '_', regex=False)
X_test.columns = X_test.columns.str.replace(' ', '_', regex=False)

# Oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

class_counts = pd.Series(y_resampled).value_counts()
print("Number of elements:\n", class_counts)
class_proportions = class_counts / len(y_resampled)
print("Percentages:\n", class_proportions)

# Riduzione test-set
sample_size = 1000000
X_reduced = X_resampled.sample(n=sample_size, random_state=42)
y_reduced = y_resampled.loc[X_reduced.index]

# Riassegno index train-set
X_reduced = X_reduced.reset_index(drop=True)
y_reduced = y_reduced.reset_index(drop=True)

# Riassegno valori variabili a X_train, y_train
del X_train, y_train
X_train = X_reduced
y_train = y_reduced

class_counts = pd.Series(y_train).value_counts()
print("\nNumber of elements (after reduction):\n", class_counts)
class_proportions = class_counts / len(y_train)
print("Percentages (after reduction):\n", class_proportions)


def convert_dtypes(data):
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if pd.api.types.is_float_dtype(data[col]):
                data[col] = data[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(data[col]):
                data[col] = data[col].astype(np.int32)
    elif isinstance(data, pd.Series):
        if pd.api.types.is_float_dtype(data):
            data = data.astype(np.float32)
        elif pd.api.types.is_integer_dtype(data):
            data = data.astype(np.int32)
    return data

X_train = convert_dtypes(X_train)
y_train = convert_dtypes(y_train)

X_test = convert_dtypes(X_test)
y_test = convert_dtypes(y_test)

# Algoritmi per l'apprendimento supervisionato
from supervisedLearning import randomForest, gradientBoosting, logisticRegression
randomForest(X_train, X_test, y_train, y_test)
gradientBoosting(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)