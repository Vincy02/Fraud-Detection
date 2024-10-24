import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def analizeAmountSuspicious(data):
	print(np.mean(data['transaction_amount']))
	print(np.max(data['transaction_amount']))
	print(np.min(data['transaction_amount']))
	thresholds = {0: 0, 500: 0, 1000: 0, 2000: 0, 5000: 0, 10000: 0}
	for index, row in data.iterrows():
		transaction_amount = row['transaction_amount']
		for threshold, count in sorted(thresholds.items(), reverse=True):
			if transaction_amount >= threshold:
				thresholds[threshold] += 1
				break
	print(thresholds)

def analizeTreshold(data):
	product_category = {'clothing':[], 'home & garden':[], 'toys & games':[], 'health & beauty':[], 'electronics':[]}
	for index, row in data.iterrows():
	    category = row['product_category']
	    if category in product_category:
	        product_category[category].append(row['quantity'])

	average_product_category = {category: np.mean(amounts) for category, amounts in product_category.items()}
	max_product_category = {category: np.max(amounts) for category, amounts in product_category.items()}
	min_product_category = {category: np.min(amounts) for category, amounts in product_category.items()}

	print(average_product_category)
	print(max_product_category)
	print(min_product_category)

def analizeData(data):
	# Controllo ditribuzione dei dati
	fraud_counts = data['is_fraudulent'].value_counts()

	# Istogramma e grafico a torta della distribuzione dei dati
	plt.figure(figsize=(6,3))
	fraud_counts.plot(kind='bar', color=['skyblue', 'salmon'])
	plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
	plt.xlabel('0: Legit, 1: Fraudulent')
	plt.ylabel('Number of transaction')
	plt.show()

	plt.figure(figsize=(7, 7))
	plt.pie(fraud_counts, labels=['Legit', 'Fraudulent'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
	plt.title('Percentage of Fraudulent and Non-Fraudulent Transactions')
	plt.show()

	# Grafici a torta per la distribuzione del device utilizzato
	data_grouped = data.groupby(['device_used', 'is_fraudulent']).size().reset_index(name='count')
	data_fraud = data_grouped[data_grouped['is_fraudulent'] == 1]
	data_not_fraud = data_grouped[data_grouped['is_fraudulent'] == 0]

	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.pie(data_fraud['count'], labels=data_fraud['device_used'], autopct='%1.1f%%')
	plt.title('Device used in fraud transaction')

	plt.subplot(1, 2, 2)
	plt.pie(data_not_fraud['count'], labels=data_not_fraud['device_used'], autopct='%1.1f%%')
	plt.title('Device used in non-fraud transaction')

	plt.show()

	# Grafici a torta per distribuzione di metodo di pagamento per transazioni fraudolente e non
	data_grouped = data.groupby(['payment_method', 'is_fraudulent']).size().reset_index(name='count')
	data_fraud = data_grouped[data_grouped['is_fraudulent'] == 1]
	data_not_fraud = data_grouped[data_grouped['is_fraudulent'] == 0]	

	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.pie(data_fraud['count'], labels=data_fraud['payment_method'], autopct='%1.1f%%')
	plt.title('Payment method in fraud transaction')

	plt.subplot(1, 2, 2)
	plt.pie(data_not_fraud['count'], labels=data_not_fraud['payment_method'], autopct='%1.1f%%')
	plt.title('Payment method in non-fraud transaction')

	plt.show()

	# Istogramma che mostra la distribuzione del numero di transazioni nel tempo
	data['transaction_date'] = pd.to_datetime(data['transaction_date']).dt.date
	#data = data[data['customer_location']=='East Michael']
	data_grouped = data.groupby(['transaction_date', 'is_fraudulent']).size().reset_index(name='count')

	sns.lineplot(x='transaction_date', y='count', hue='is_fraudulent', data=data_grouped)
	plt.title('Number of transactions over time')
	plt.xlabel('Transaction date')
	plt.ylabel('Number of transaction')
	plt.show()

	# Kernel density plot
	data['transaction_date'] = pd.to_datetime(data['transaction_date'])
	data_fraud = data[data['is_fraudulent'] == 1]
	data_not_fraud = data[data['is_fraudulent'] == 0]

	sns.kdeplot(data=data_fraud, x='transaction_date', label='Fraudulent')
	sns.kdeplot(data=data_not_fraud, x='transaction_date', label='Non Fraudulent')

	plt.title('Transaction density over time')
	plt.xlabel('Transaction date')
	plt.ylabel('Density')
	plt.show()

	# Calcolo della matrice di correlazione
	corr_matrix = data[['transaction_amount', 'quantity', 'customer_age','account_age_days', 'is_fraudulent']].corr()

	# Plot della heatmap
	plt.figure(figsize=(10, 8))
	sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
	plt.title('Correlation Matrix among Numerical Variables')
	plt.show()