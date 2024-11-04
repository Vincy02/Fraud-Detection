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