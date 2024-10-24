#from installLibreries import installPackages
#installPackages()

import inflection
import pandas as pd
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

'''
# Analisi dei dati a disposizione
from datasetAnalysis import analizeData, analizeTreshold, analizeAmountSuspicious
analizeAmountSuspicious(data.copy())
analizeTreshold(data.copy())
analizeData(data.copy())
'''

# Creazione della KB in prolog
from prolog import create_kb, define_rules, consult_kb
create_kb(data.copy())
define_rules()
consult_kb(data.copy())

# Importazioni nuovo dataset
data = pd.read_csv("../Data/new_data.csv")

# Rimozione della feature non importanti (v2)
data.drop(['transaction_id', 'customer_id', 'shipping_address', 'billing_address',
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

# Inizio divisione dataset: train - test e normalizzazione dei dati
X = data[categorical_columns + numeric_columns].copy()
y = data['is_fraudulent'].copy()

'''
# Apprendimento non supervisionato
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

X = pd.get_dummies(X, columns=categorical_columns)
X.columns = X.columns.str.replace(' ', '_', regex=False)

from unSupervisedLearning import calcolaCluster
etichette_cluster, centroidi, silhouette_avg = calcolaCluster(X)
# X['clusterIndex'] = etichette_cluster
# X.to_csv("../Data/unSup.csv", index=False)
'''

# Apprendimento supervisionato - preparazione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dati numerici
scaler = StandardScaler()
scaler.fit(X_train[numeric_columns])
X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# One-hot encoder su dati categorici
X_train = pd.get_dummies(X_train, columns=categorical_columns)
X_test = pd.get_dummies(X_test, columns=categorical_columns)
X_train.columns = X_train.columns.str.replace(' ', '_', regex=False)
X_test.columns = X_test.columns.str.replace(' ', '_', regex=False)

# Algoritmi per l'apprendimento supervisionato
from supervisedLearning import randomForest, gradientBoosting

randomForest(X_train, X_test, y_train, y_test)

gradientBoosting(X_train, X_test, y_train, y_test)