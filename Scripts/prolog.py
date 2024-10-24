from pyswip import Prolog
import datetime

kb_path = "../Data/"
kb_file_name = "kb.pl"
rules_file_name = "rules.pl"

def create_kb(data):
	with open(kb_path+kb_file_name, 'w') as f:
		for index, row in data.iterrows():
			transaction_id = row['transaction_id']
			customer_id = row['customer_id']
			transaction_amount = row['transaction_amount']
			transaction_date = row['transaction_date']
			transaction_date, _aux = transaction_date.split()
			transaction_date = datetime.datetime.strptime(transaction_date, "%Y-%m-%d")
			start_date = datetime.datetime(2024, 1, 1)
			date = (transaction_date - start_date).days
			payment_method = row['payment_method']
			product_category = row['product_category']
			quantity = row['quantity']
			customer_age = row['customer_age']
			customer_location = row['customer_location']
			device_used = row['device_used']
			shipping_address = row['shipping_address'].replace('\n', ' ')
			billing_address = row['billing_address'].replace('\n', ' ')
			is_fraudulent = row['is_fraudulent']
			account_age_days = row['account_age_days']
			transaction_hour = row['transaction_hour']
			f.write(f"transaction('{transaction_id}', '{customer_id}', {transaction_amount}, {date}, '{payment_method}', '{product_category}', {quantity}, {customer_age}, '{customer_location}', '{device_used}', '{shipping_address}', '{billing_address}', {is_fraudulent}, {account_age_days}, {transaction_hour}).\n")

def define_rules():
	with open(kb_path+rules_file_name, 'w') as f:
		# Definizione delle soglie per le categorie
		suspicious_threshold_1 = "suspicious_threshold('clothing', 4).\n"
		f.write(suspicious_threshold_1)

		suspicious_threshold_2 = "suspicious_threshold('home & garden', 4).\n"
		f.write(suspicious_threshold_2)

		suspicious_threshold_3 = "suspicious_threshold('toys & games', 4).\n"
		f.write(suspicious_threshold_3)

		suspicious_threshold_4 = "suspicious_threshold('health & beauty', 4).\n"
		f.write(suspicious_threshold_4)

		suspicious_threshold_5 = "suspicious_threshold('electronics', 4).\n"
		f.write(suspicious_threshold_5)

		# Regole
		rule_suspicious_transaction = "is_suspicious_transaction(TransactionID) :- transaction(TransactionID, _, Amount, _, PaymentMethod, _, _, _, _, _, _, _, _, _, _), (Amount >= 2000, PaymentMethod == 'bank transfer').\n"
		f.write(rule_suspicious_transaction)

		rule_high_risk_category_and_amount = "high_risk_category_and_amount(TransactionID) :- transaction(TransactionID, _, _, _, _, Category, Quantity, _, _, _, _, _, _, _, _), suspicious_threshold(Category, Threshold), Quantity > Threshold.\n"
		f.write(rule_high_risk_category_and_amount)

		suspicious_customer_location = """suspicious_customer_location(TransactionID, CustomerLocation) :-
    	transaction(TransactionID, _, _, Date, _, _, _, _, CustomerLocation, _, _, _, _, _, _),
    	findall(_Date, (
      		transaction(_TransactionID, _, _, _Date, _, _, _, _, CustomerLocation, _, _, _, _, _, _),
        	_TransactionID \\= TransactionID,
        	Date1 is Date,
        	Date2 is _Date,
        	Diff is abs(Date1 - Date2),
        	Diff =< 2
    	), Days),
    	length(Days, Count),
    	Count >= 50.\n"""
		f.write(suspicious_customer_location)

def consult_kb(data):
	prolog = Prolog()
	prolog.consult(kb_path+kb_file_name)
	prolog.consult(kb_path+rules_file_name)

	suspicious_transactions = []
	high_risk_transactions = []
	suspicious_customer_location = []

	for index, row in data.iterrows():
		transaction_id = row['transaction_id']
		customer_location = row['customer_location']

		# Eseguo regola is_suspicious_transaction
		if list(prolog.query(f"is_suspicious_transaction('{transaction_id}')")):
			suspicious_transactions.append(1)
		else:
			suspicious_transactions.append(0)

		# Eseguo regola high_risk_category_and_amount
		if list(prolog.query(f"high_risk_category_and_amount('{transaction_id}')")):
			high_risk_transactions.append(1)
		else:
			high_risk_transactions.append(0)

		# Eseguo regola suspicious_customer_location
		if list(prolog.query(f"suspicious_customer_location('{transaction_id}', '{customer_location}')")):
			suspicious_customer_location.append(1)
		else:
			suspicious_customer_location.append(0)

	# Aggiunta delle nuove colonne al dataframe
	data['is_suspicious_transaction'] = suspicious_transactions
	data['high_risk_category_and_amount'] = high_risk_transactions
	data['suspicious_customer_location'] = suspicious_customer_location

	data.to_csv('../Data/new_data.csv', index=False)