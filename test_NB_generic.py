from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from NB_generic import *


def test_crossvalidation_sklearn(X, y):
    cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=44)

    print(f"Results for GaussianNB from sklearn:")
    sensitivity_values = []
    specificity_values = []
    accuracy_values = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        conf_matrix, accuracy, sensitivity, specificity = calculate_confusion_matrix(
            y_test, y_predict
        )

        accuracy_values.append(accuracy)
        sensitivity_values.append(sensitivity)
        specificity_values.append(specificity)

    mean_accuracy = np.mean(accuracy_values, axis=0)
    mean_sensitivity = np.mean(sensitivity_values, axis=0)
    mean_specificity = np.mean(specificity_values, axis=0)

    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Sensitivity: {np.round(mean_sensitivity, 4)}")
    print(f"Mean Specificity: {np.round(mean_specificity, 4)}\n")


if __name__ == "__main__":
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features, columns=iris.data.feature_names)
    y = pd.DataFrame(iris.data.targets, columns=["class"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=43
    )
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=["class"]).values.ravel()
    y_test = pd.DataFrame(y_test, columns=["class"]).values.ravel()

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("======= DISTRIBUTION COMPARISON TEST =======")
    print("Sklearn Naive Bayes classification accuracy", np.mean(y_test == y_pred))
    print("\n======= CROSS-VALIDATION TEST =======")
    test_crossvalidation_sklearn(X, y)
