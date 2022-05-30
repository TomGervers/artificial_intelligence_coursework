import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dadta
data = pd.read_csv('data.csv')

# Split the data in train and test set
Xs = data.drop(columns=['cell_id', 'most_present_age'])
y = data['most_present_age']
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3,
                                                    random_state=0)

# Decision tree using gini criterion
dtc = DecisionTreeClassifier(criterion='gini')
clf = dtc.fit(X_train, y_train)
# Accuracy score on train and test set
clf.score(X_train, y_train)
clf.score(X_test, y_test)

# Decision tree using entropy
dtc = DecisionTreeClassifier(criterion='entropy')
clf = dtc.fit(X_train, y_train)
clf.score(X_test, y_test)
# Accuracy score on train and test set
clf.score(X_train, y_train)
clf.score(X_test, y_test)

# print the ten most important features for a fitted tree
print('Most important to least important features')
print('------------------------------------------')
for importance, name in sorted(zip(clf.feature_importances_, X_train.columns),
                               reverse=True)[:]:
    print(name, importance, '\n')
print('------------------------------------------')


plt.figure(figsize=(50, 50))  # set plot size (denoted in inches)
plot_tree(clf, fontsize=10)
plt.show()
