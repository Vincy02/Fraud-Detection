import pandas as pd
import numpy as np
import cuml
import dask_ml.model_selection as dcv
from scipy.stats import randint, uniform
import time
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from resultsAnalysis import plot_and_evaluate

class Timer:
	def __enter__(self):
		self.tick = time.time()
		return self

	def __exit__(self, *args, **kwargs):
		self.tock = time.time()
		self.elapsed = self.tock - self.tick

def evaluate_model(model, X_train, X_test, y_train, y_test, file_name):
	original_stdout = sys.stdout

	if os.path.exists(file_name+".txt"):
		os.remove(file_name+".txt")

	with open(file_name+".txt", 'a') as f:
		sys.stdout = f

		with Timer() as crf_fit_time:
			model.fit(X_train, y_train)
		print("> Time elapsed (fit):", crf_fit_time.elapsed)

		best_rf_model = model.best_estimator_

		with Timer() as crf_predict_time:
			y_pred_rf = best_rf_model.predict(X_test)
		print("> Time elapsed (predict):", crf_predict_time.elapsed)

		with Timer() as crf_scores_time:
			accuracy_rf = accuracy_score(y_test, y_pred_rf)
			precision_rf = precision_score(y_test, y_pred_rf)
			recall_rf = recall_score(y_test, y_pred_rf)
			f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
		print("> Time elapsed (calc scores):", crf_scores_time.elapsed)

		print("Best Hyperparameters:", model.best_params_)
		print("Accuracy (TEST / EVAL):", accuracy_rf)
		print("Precision (TEST / EVAL):", precision_rf)
		print("Recall (TEST / EVAL):", recall_rf)
		print("F1 (TEST / EVAL):", f1_rf)
	sys.stdout = original_stdout
	cv_results_df = pd.DataFrame(model.cv_results_)
	cv_results_df.to_csv('%s.csv' % file_name, index=False)
	plot_and_evaluate(file_name)

def randomForest(X_train, X_test, y_train, y_test):
	print("Start Random Forest")
	np.random.seed(42)
	params = {
		'n_estimators': randint(low=10, high=300),
		'max_depth': randint(low=4, high=12),
		'min_samples_split': randint(low=2, high=10),
		'min_samples_leaf': randint(low=1, high=8)
	}

	name = 'RandomForestClassifier'
	crf = dcv.RandomizedSearchCV(estimator=cuml.ensemble.RandomForestClassifier(),
								 param_distributions=params,
								 n_iter=50,
								 cv=4,
								 return_train_score=True,
								 random_state=42)
	evaluate_model(crf, X_train, X_test, y_train, y_test, "../Results/RandomForestClassifier/"+name)


def gradientBoosting(X_train, X_test, y_train, y_test):
	print("Start Gradient Boosting")
	np.random.seed(42)
	params = {
		'n_estimators': randint(low=100, high=300),
		'learning_rate': [0.1, 0.15, 0.2],
		'max_depth': randint(low=4, high=12),
		'subsample': [0.5, 0.7, 1.0]
	}
	name = 'GradientBoostingClassifier'
	import xgboost as xgb
	cgb = dcv.RandomizedSearchCV(estimator=xgb.XGBClassifier(device = "cuda", eval_metric='logloss'),
								 param_distributions=params,
								 n_iter=50,
								 cv=4,
								 return_train_score=True,
								 random_state=42)
	evaluate_model(cgb, X_train, X_test, y_train, y_test, "../Results/GradientBoostingClassifier/"+name)

def logisticRegression(X_train, X_test, y_train, y_test):
	print("Start Logistic Regression")
	params = {
		'penalty' :['l2'],
		'C': uniform(0.01, 10),
		'max_iter': [1000, 10000, 15000]
	}
	name = 'LogisticRegression'
	from sklearn.model_selection import RandomizedSearchCV
	clr = RandomizedSearchCV(estimator=cuml.linear_model.LogisticRegression(),
							 param_distributions=params,
							 n_iter=50,
							 cv=4,
							 return_train_score=True,
							 random_state=42)
	evaluate_model(clr, X_train, X_test, y_train, y_test, "../Results/LogisticRegression/"+name)