import pandas as pd
import sys
import matplotlib.pyplot as plt

def plot_and_evaluate(file_name):
	data = pd.read_csv(file_name+".csv")

	data = data[['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'mean_test_score','std_test_score',
				 'split0_train_score','split1_train_score','split2_train_score','split3_train_score','mean_train_score','std_train_score']]

	iterations = range(1, len(data) + 1)

	# Plot di Mean Test Score
	plt.plot(iterations, data['mean_test_score'], "-", color='b', label='Mean Test Score')
	plt.fill_between(iterations,
					 data['mean_test_score'] - data['std_test_score'],
					 data['mean_test_score'] + data['std_test_score'],
					 color='b', alpha=0.2)

	# Plot di Mean Train Score
	plt.plot(iterations, data['mean_train_score'], "--", color='g', label='Mean Train Score')
	plt.fill_between(iterations,
					 data['mean_train_score'] - data['std_train_score'],
					 data['mean_train_score'] + data['std_train_score'],
					 color='g', alpha=0.2)

	# Evidenzio best model
	best_iteration = data['mean_test_score'].idxmax()
	best_score = data['mean_test_score'].max()
	plt.scatter(best_iteration+1, best_score, color='r', s=100, zorder=5, label='Best Model')

	plt.title('Learning Curve')
	plt.xlabel('Iteration')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid(True)
	plt.legend(loc="best")
	plt.savefig('%s_learning_curve.png' % file_name)

	if os.path.exists(file_name+".txt"):
		os.remove(file_name+".txt")

	original_stdout = sys.stdout
	with open(file_name+".txt", 'a') as f:
		sys.stdout = f

		row = data.loc[best_iteration]

		print("Accuracy (Train): ", row['mean_train_score'], "\t|\tAccuracy (Test-set CV): ", row['mean_test_score'], "\t(Best Model)")
		print("Standard Deviation:", row['std_train_score'], "\t|\tVariance:", row['std_train_score']*row['std_train_score'], "\t(Train-Set | Best Model)")
		print("Standard Deviation:", row['std_test_score'], "\t|\tVariance:", row['std_test_score']*row['std_test_score'], "\t(Test-Set CV | Best Model)")

		diff_mean = row['mean_test_score'] - row['mean_train_score']
		print("Difference Mean Score (Test CV - Train) - best model:", diff_mean)
	sys.stdout = original_stdout

'''
# da rivedere #
potrei vedere differenze tra i diversi modelli con test statistico

differences = test_score_modello_1 - test_score_modello_2 ex.

_, p_value_normality = stats.shapiro(differences)
print(p_value_normality)

t_statistic, p_value = stats.ttest_rel(data['mean_train_score'], data['mean_test_score'])
print(p_value)
'''

def plot_and_evaluate_NN(file_name):
	data = pd.read_csv(file_name+".csv")

	# Find the best epoch based on highest test accuracy
	best_epoch_idx = data['mean_test_accuracy'].idxmax()
	best_epoch = data['epoch'][best_epoch_idx]

	# Retrieve all metrics for the best epoch
	best_metrics = {
		"Epoch": best_epoch,
		"Train Accuracy": data['mean_train_accuracy'][best_epoch_idx],
		"Test Accuracy": data['mean_test_accuracy'][best_epoch_idx],
		"Train Precision": data['mean_train_precision'][best_epoch_idx],
		"Test Precision": data['mean_test_precision'][best_epoch_idx],
		"Train Recall": data['mean_train_recall'][best_epoch_idx],
		"Test Recall": data['mean_test_recall'][best_epoch_idx],
		"Train F1 Score": data['mean_train_f1'][best_epoch_idx],
		"Test F1 Score": data['mean_test_f1'][best_epoch_idx]
	}

	original_stdout = sys.stdout
	with open(file_name+".txt", 'a') as f:
		sys.stdout = f
		print("Best Model (based on highest test accuracy):")
		for metric, value in best_metrics.items():
			print(f"{metric}: {value:.4f}")
	sys.stdout = original_stdout

	# Plotting
	plt.figure(figsize=(12, 8))

	# Accuracy plot
	plt.subplot(2, 2, 1)
	plt.plot(data['epoch'], data['mean_train_accuracy'], label='Train Accuracy', marker='o')
	plt.plot(data['epoch'], data['mean_test_accuracy'], label='Test Accuracy', marker='o')
	plt.scatter(best_epoch, best_metrics["Test Accuracy"], color='red', label='Best Model (Test Accuracy)', zorder=5)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy over Epochs')
	plt.legend()

	# Precision plot
	plt.subplot(2, 2, 2)
	plt.plot(data['epoch'], data['mean_train_precision'], label='Train Precision', marker='o')
	plt.plot(data['epoch'], data['mean_test_precision'], label='Test Precision', marker='o')
	plt.xlabel('Epoch')
	plt.ylabel('Precision')
	plt.title('Precision over Epochs')
	plt.legend()

	# Recall plot
	plt.subplot(2, 2, 3)
	plt.plot(data['epoch'], data['mean_train_recall'], label='Train Recall', marker='o')
	plt.plot(data['epoch'], data['mean_test_recall'], label='Test Recall', marker='o')
	plt.xlabel('Epoch')
	plt.ylabel('Recall')
	plt.title('Recall over Epochs')
	plt.legend()

	# F1 Score plot
	plt.subplot(2, 2, 4)
	plt.plot(data['epoch'], data['mean_train_f1'], label='Train F1 Score', marker='o')
	plt.plot(data['epoch'], data['mean_test_f1'], label='Test F1 Score', marker='o')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.title('F1 Score over Epochs')
	plt.legend()

	plt.tight_layout()
	plt.savefig('%s_learning_curve.png' % file_name)