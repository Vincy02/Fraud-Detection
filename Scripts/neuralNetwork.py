import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def NNBinaryClassifier(X_train, X_test, y_train, y_test):
	# Conversione tipi dati -> float32
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	y_train = y_train.astype('float32').values.reshape(-1, 1)
	y_test = y_test.astype('float32').values.reshape(-1, 1)

	# Creazione tensor PyTorch
	X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

	# Creazione dei dataset e dei dataloader
	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	# Definizione modello
	class BinaryClassifier(nn.Module):
		def __init__(self, input_size, hidden_size):
			super(BinaryClassifier, self).__init__()
			self.fc1 = nn.Linear(input_size, hidden_size)
			self.relu = nn.ReLU()
			self.dropout = nn.Dropout(0.2)
			self.fc2 = nn.Linear(hidden_size, 1)
			self.sigmoid = nn.Sigmoid()

		def forward(self, x):
			out = self.fc1(x)
			out = self.relu(out)
			out = self.dropout(out)
			out = self.fc2(out)
			out = self.sigmoid(out)
			return out

	# Iperparametri
	input_size = X_train.shape[1]
	hidden_size = 64
	num_epochs = 5
	learning_rate = 0.01

	best_val_accuracy = 0.0
	best_model_state_dict = None

	# Inizializzazione
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = BinaryClassifier(input_size, hidden_size).to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	mean_train_accuracies, mean_train_precisions, mean_train_recalls, mean_train_f1s = [], [], [], []
	mean_test_accuracies, mean_test_precisions, mean_test_recalls, mean_test_f1s = [], [], [], []

	# Addestramento
	for epoch in range(num_epochs):
		train_accuracies, train_precisions, train_recalls, train_f1s = [], [], [], []
		test_accuracies, test_precisions, test_recalls, test_f1s = [], [], [], []
		for i, (inputs, labels) in enumerate(train_loader):
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			if (i+1) % 1000 == 0:
				model.eval()
				with torch.no_grad():
					train_inputs, train_labels = next(iter(train_loader))
					train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
					train_outputs = model(train_inputs)
					train_predicted = (train_outputs > 0.5).float()
					train_accuracy = accuracy_score(train_labels.cpu().numpy(), train_predicted.cpu().numpy())
					train_precision = precision_score(train_labels.cpu().numpy(), train_predicted.cpu().numpy())
					train_recall = recall_score(train_labels.cpu().numpy(), train_predicted.cpu().numpy())
					train_f1 = f1_score(train_labels.cpu().numpy(), train_predicted.cpu().numpy())
					train_accuracies.append(train_accuracy)
					train_precisions.append(train_precision)
					train_recalls.append(train_recall)
					train_f1s.append(train_f1)

					test_inputs, test_labels = next(iter(test_loader))
					test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
					test_outputs = model(test_inputs)
					test_predicted = (test_outputs > 0.5).float()
					test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predicted.cpu().numpy())
					test_precision = precision_score(test_labels.cpu().numpy(), test_predicted.cpu().numpy())
					test_recall = recall_score(test_labels.cpu().numpy(), test_predicted.cpu().numpy())
					test_f1 = f1_score(test_labels.cpu().numpy(), test_predicted.cpu().numpy())
					test_accuracies.append(test_accuracy)
					test_precisions.append(test_precision)
					test_recalls.append(test_recall)
					test_f1s.append(test_f1)
					print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]")
			model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}] ended.")
		print(f"[Train]\tMean Accuracy: {np.mean(train_accuracies):.4f}\tMean Precision: {np.mean(train_precisions):.4f}\tMean Recall: {np.mean(train_recalls):.4f}\tMean F1: {np.mean(train_f1s):.4f}")
		print(f"[Test]\tMean Accuracy: {np.mean(test_accuracies):.4f}\tMean Precision: {np.mean(test_precisions):.4f}\tMean Recall: {np.mean(test_recalls):.4f}\tMean F1: {np.mean(test_f1s):.4f}")
		mean_train_accuracies.append(np.mean(train_accuracies))
		mean_train_precisions.append(np.mean(train_precisions))
		mean_train_recalls.append(np.mean(train_recalls))
		mean_train_f1s.append(np.mean(train_f1s))
		mean_test_accuracies.append(np.mean(test_accuracies))
		mean_test_precisions.append(np.mean(test_precisions))
		mean_test_recalls.append(np.mean(test_recalls))
		mean_test_f1s.append(np.mean(test_f1s))

		# Trovo best model
		val_accuracy = np.mean(test_accuracies)
		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			best_model_state_dict = model.state_dict()
			print(f"New best model found with accuracy: {best_val_accuracy:.4f}")

	# Salvo best model individuato (best acc)
	torch.save(best_model_state_dict, "../Results/NNBinaryClassifier/best_model.pth")

	data = {'epoch': list(range(1, num_epochs+1)),
		'mean_train_accuracy': mean_train_accuracies,
		'mean_train_precision': mean_train_precisions,
		'mean_train_recall': mean_train_recalls,
		'mean_train_f1': mean_train_f1s,
		'mean_test_accuracy': mean_test_accuracies,
		'mean_test_precision': mean_test_precisions,
		'mean_test_recall': mean_test_recalls,
		'mean_test_f1': mean_test_f1s
	}
	df = pd.DataFrame(data)

	# Salvo risultati
	df.to_csv("../Results/NNBinaryClassifier/NNBinaryClassifier.csv", index=False)

	from resultsAnalysis import plot_and_evaluate_NN
	plot_and_evaluate_NN("../Results/NNBinaryClassifier/NNBinaryClassifier")