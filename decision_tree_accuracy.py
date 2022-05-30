import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dadta
data = pd.read_csv('data.csv')

# Split the data into train and test set
Xs = data.drop(columns=['cell_id', 'most_present_age'])
y = data['most_present_age']
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3,
                                                    random_state=0)

# Calculate accuracy for varying tree depth and see how it changes
dtc = DecisionTreeClassifier()
scores = []
max_score = 0.0
best_depth = 0

for dep in range(1, 30):

    dtc = DecisionTreeClassifier(max_depth=dep)
    clf = dtc.fit(X_train, y_train)
    scores.append([dep, clf.score(X_test, y_test)])
    temp = clf.score(X_test, y_test)
    if temp > max_score:
        max_score = temp
        best_depth = dep

"""
NO MAX DEPTH
"""
# Plot confusion matrix on predictions
dtc = DecisionTreeClassifier(criterion='entropy')
clf = dtc.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

# Plot heatmap of results
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f",
            linewidths=.5, square=True, cmap='rocket_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.3f}'.format(clf.score(X_test, y_test))
plt.title(all_sample_title, size=15)
plt.show()

# Print accuracy with no max depth
print('Accuracy: %f achieved with no max depth' % (clf.score(X_test, y_test)))

# Plot decision tree with no max depth
plt.figure(figsize=(50, 50))
plot_tree(clf, fontsize=10)
plt.show()

"""
BEST DEPTH
"""
# Plot confusion matrix on predictions
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=(best_depth))
clf = dtc.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

# Plot heatmap of results
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f",
            linewidths=.5, square=True, cmap='rocket_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score at best depth: {:.3f}'.format(
    clf.score(X_test, y_test))
plt.title(all_sample_title, size=15)
plt.show()

# Print accuracy at best depth
print('Best accuracy: %f achieved at depth: %d' % (max_score, best_depth))

# Plot decision tree to best depth
plt.figure(figsize=(50, 50))
plot_tree(clf, fontsize=10)
plt.show()

# Print full list of accuracy
print('\nFull list of accuracy at depths')
print('--------------------------------')
for s in scores:
    print('Depth:', *s)
print('--------------------------------')
