## Decision Tree Classification on the Iris Dataset

This repository contains an implementation of a Decision Tree Classifier using the Iris dataset. The model classifies flowers into three species based on four features: sepal length, sepal width, petal length, and petal width. The classifier is trained on a portion of the dataset and evaluated on a separate test set. Additionally, the trained decision tree is visualized to show how it makes decisions.

## Libraries Used
Numpy: For numerical computations and array manipulation.
Pandas: For data manipulation (though not directly used in the code, it can be useful for dataset handling).
Scikit-learn (sklearn):
For loading the Iris dataset.
For splitting the data into training and test sets.
For creating and evaluating the Decision Tree Classifier.
Matplotlib: For visualizing the trained Decision Tree.
## Dataset
The Iris dataset is used, which contains 150 samples of Iris flowers, each belonging to one of three species:
Setosa,
Versicolor,
Virginica
The dataset consists of 4 features:
Sepal length,
Sepal width,
Petal length,
Petal width.

## Steps
# 1. Load the Dataset
The Iris dataset is loaded using sklearn.datasets.load_iris().

# 2. Split the Data
The dataset is divided into training and testing sets with an 80/20 split using train_test_split().

# 3. Train the Model
A Decision Tree Classifier (DecisionTreeClassifier) is trained using the training data.

# 4. Evaluate the Model
The model is evaluated on the test set, and the accuracy is computed using accuracy_score(). The accuracy is then printed as a percentage.

# 5. Visualize the Decision Tree
The trained Decision Tree is visualized using plot_tree() from sklearn.tree, showing the splits, features, and the decision-making process of the tree.
