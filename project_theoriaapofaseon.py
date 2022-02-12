import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import validation_curve
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def svm(X_train,X_test ,y_train,  y_test):

    svm_classifier =  SVC(kernel= 'rbf')
    svm_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = svm_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = svm_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    print('\nAccuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    title = "Confusion matrix, without normalization for SVM"
    disp = ConfusionMatrixDisplay.from_estimator(
            svm_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues
        )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()

    return svm_classifier


def Naive_Bayes(X,y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred =nb_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = nb_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    title = "Confusion matrix, without normalization for Naive Bayes"

    disp = ConfusionMatrixDisplay.from_estimator(
            nb_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues
        )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()

    print('\nAccuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return nb_classifier


def Logistic_Regresion(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lr_classifier = LogisticRegression(max_iter=1000)
    lr_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = lr_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = lr_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Logistic Regression"

    disp = ConfusionMatrixDisplay.from_estimator(
            lr_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues,
        )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()

    print('\nAccuracy for training set for Logistic Regression = {}'.format(
        (cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return lr_classifier


def Decision_Tree(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = dt_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = dt_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Decision Tree"
    disp = ConfusionMatrixDisplay.from_estimator(
            dt_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues,
        )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()


    print('\nAccuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return dt_classifier


def Random_Forest(X, y):
    # Calculate X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    rf_classifier = RandomForestClassifier(n_estimators=10)
    rf_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = rf_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = rf_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for Random Forest"

    disp = ConfusionMatrixDisplay.from_estimator(
            rf_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues
        )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()


    print('\nAccuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(random_state=1)
    # define search space
    space = dict()
    space['n_estimators'] = [10, 100, 500]
    space['max_features'] = [2, 4, 6]
    # define search
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # execute the nested cross-validation
    scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    return rf_classifier


def XGBoost(X_train,X_test, y_train,y_test):

    # X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

    xg = XGBClassifier(use_label_encoder=False)
    xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = xg.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:  # setting threshold to .5
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for XGBoost"

    disp = ConfusionMatrixDisplay.from_estimator(
            xg,
            X_test,
            y_test,
            cmap=plt.cm.Blues
        )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()


    print('\nAccuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return xg

def Kneighbors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    knn_classifier = KNeighborsClassifier(n_neighbors = 10)
    knn_classifier.fit(X_train,y_train)


    # Predicting the Test set results
    y_pred = knn_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = knn_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    title = "Confusion matrix, without normalization for KNN"
    disp = ConfusionMatrixDisplay.from_estimator(
            knn_classifier,
            X_test,
            y_test,
            cmap=plt.cm.Blues,
        )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()

    # Setting the range for the parameter (from 1 to 10)
    parameter_range = np.arange(1, 10, 1)

    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                               param_name = "n_neighbors",
                                               param_range = parameter_range,
                                               cv = 5, scoring = "accuracy")
    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
         label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score,
         label = "Cross Validation Score", color = 'g')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()



    print('\nAccuracy for training set for Kneighbors = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Kneighbors = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

    return knn_classifier

def main():

    df = pd.read_csv("Dataset 1.csv", delimiter=",")
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']
    df.isnull().sum()

    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['sex'] = df.sex.map({0: 'female', 1: 'male'})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    #plots


    sns.set_context("paper", font_scale = 1, rc = {"font.size": 18,"axes.titlesize": 20,"axes.labelsize": 20})
    sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
    plt.title('Variation of Age for each target class')
    plt.show()

    # barplot of age vs sex with hue = target
    sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
    plt.title('Distribution of age vs sex with the target class')
    plt.show()

    df['sex'] = df.sex.map({'female': 0, 'male': 1})

    # data preprocessing
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #  SVM
    svm_classifier = svm(X_train, X_test, y_train,  y_test)


    #  Naive Bayes
    nv_classifier = Naive_Bayes(X, Y)

    #  Logistic Regression
    lr_classifier = Logistic_Regresion(X, Y)

    #  Decision Tree
    dt_classifier = Decision_Tree(X, Y)

    # Random Forest
    rf_classifier = Random_Forest(X,Y)

    # applying XGBoost
    xgb_classifier = XGBoost(X_train, X_test, y_train, y_test)

    #Kneighbors Classifier
    kn_classifier = Kneighbors(X, Y)

    test = [[100, 1, 3, 160, 250, 1, 0, 180, 0, 4, 0, 0, 3],
            [100, 1, 3, 180, 300, 1, 0, 200, 0, 3, 0, 0, 3]]

    test = sc.fit_transform(test)
    #Test models for 2 cases
    for i in range(len(test)):
        print(f"Patient {i} is: \n")
        print(f"\t SVM: {svm_classifier.predict([test[i]])}" )
        print(f"\t Naive Bayes: {nv_classifier.predict([test[i]])}")
        print(f"\t Logistic Regression: {lr_classifier.predict([test[i]])}")
        print(f"\t Decision Tree: {dt_classifier.predict([test[i]])}")
        print(f"\t Random Forest: {rf_classifier.predict([test[i]])}")
        print(f"\t XGBoost: {xgb_classifier.predict([test[i]])}")
        print(f"\t KNeighbors: {kn_classifier.predict([test[i]])}")
        print()

if __name__ == "__main__":
    main()
