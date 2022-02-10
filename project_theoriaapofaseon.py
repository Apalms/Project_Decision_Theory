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
from matplotlib.cm import rainbow

from xgboost import XGBClassifier





def svm(X_train,X_test ,y_train,  y_test):
    svm_scores = []
    kernels = ['linear', 'poly', 'rbf', 'sigmoid' ]
    for i in range(len(kernels)):
        svm_classifier =  SVC(kernel= kernels[i])
        svm_classifier.fit(X_train, y_train)
        svm_scores.append(svm_classifier.score(X_test,y_test))


    # Predicting the Test set results
    y_pred = svm_classifier.predict(X_test)


    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = svm_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)


    print('\nAccuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


    colors = rainbow(np.linspace(0, 1, len(kernels)))
    plt.bar(kernels, svm_scores, color = colors)
    for i in range(len(kernels)):
        plt.text(i, svm_scores[i], svm_scores[i])
    plt.xlabel('Kernels')
    plt.ylabel('Scores')
    plt.title('Support Vector Classifier scores for different kernels')
    plt.show()


def Naive_Bayes(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

   

    print('\nAccuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def Logistic_Regresion(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print('\nAccuracy for training set for Logistic Regression = {}'.format(
        (cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def Decision_Tree(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print('\nAccuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def Random_Forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print('\nAccuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))




def XGBoost(X_train,X_test, y_train,y_test):
    # from sklearn.model_selection import train_test_split
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
    print('\nAccuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

def Kneighbors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    knn_scores = []
    for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
        knn_classifier.fit(X_train,y_train)
        knn_scores.append(knn_classifier.score(X_test,y_test))

    # Predicting the Test set results
    y_pred = knn_classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = knn_classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print('\nAccuracy for training set for Kneighbors = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for Kneighbors = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

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
    svm(X_train, X_test, y_train,  y_test)


    #  Naive Bayes
    Naive_Bayes(X, Y)

    #  Logistic Regression
    Logistic_Regresion(X, Y)

    #  Decision Tree
    Decision_Tree(X, Y)

    # Random Forest
    Random_Forest(X,Y)


    # applying XGBoost
    XGBoost(X_train, X_test, y_train, y_test)

    #Kneighbors Classifier
    Kneighbors(X, Y)


if __name__ == "__main__":
    main()
