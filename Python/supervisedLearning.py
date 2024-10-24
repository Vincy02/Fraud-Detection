from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def randomForest(X_train, X_test, y_train, y_test):
	model_rf = RandomForestClassifier()
	model_rf.fit(X_train, y_train)
	y_pred = model_rf.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy Random Forest: ", accuracy)

def gradientBoosting(X_train, X_test, y_train, y_test):
	model_gb = GradientBoostingClassifier()
	model_gb.fit(X_train, y_train)
	y_pred = model_gb.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy Gradient Boosting: ", accuracy)