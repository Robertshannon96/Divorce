from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from eda import *
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


def split(df):
    """
    Performs a 33/67 train,test split on the DataFrame.

    Input: DataFrame
    Output: X/Y train and test splits. Also returns X and y values.
    """
    X = df.drop('Class', axis = 1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=46)
    return X_train, X_test, y_train, y_test, X, y


def logistic_regression(df):
    """
    Performs Logistic Regression on a passed in data set.
    Calls the split function to split data then fits split data to model.

    **Important** to note that this logistic regression takes in a PCA DataFrame.

    Input: DataFrame
    Output: Coefficients of Logistic regression, Confusion Matrix, Model score
    """
    X_train, X_test, y_train, y_test, X, y = split(df)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Coefficients:",log_reg.coef_)  # determine most important questions
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print('Logistic Regression Accuracy: ', log_reg.score(X, y))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))


def decision_tree(df):
    """
    Applies a decision tree model to the DataFrame.

    Input: DataFrame
    Output: Accuracy Score, Confusion Matrix, visualization of tree
    """
    X_train, X_test, y_train, y_test, X, y = split(df)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True, feature_names=features, class_names=['Yes', 'No'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree_viz.png')
    Image(graph.create_png())


def main():
    df = get_data()
    pca_df = pca(df)
    logistic_regression(df=pca_df)
    decision_tree(df)


if __name__ == '__main__':
    main()




