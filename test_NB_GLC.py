from sklearn.model_selection import StratifiedKFold
from NB_GLC import *


def cauchy_curve(x, location, scale):
    return 1 / (np.pi * scale * (1 + ((x - location) / scale) ** 2))


def draw_custom_curves_cauchy(feature_stats):
    for class_label, features in feature_stats.items():
        plt.suptitle(f"Class: {class_label}")
        for feature, params in features.items():
            x = np.linspace(
                params["mean"] - 3 * params["b"], params["mean"] + 3 * params["b"], 100
            )
            y = cauchy_curve(x, params["mean"], params["b"])
            plt.plot(x, y, label=f"{feature}")
        plt.legend()
        plt.show()


def calculate_and_display_results(y_true, y_pred, distribution_type):
    conf_matrix, accuracy, sensitivity, specificity = calculate_confusion_matrix(
        y_true, y_pred
    )

    print(f"Results for {distribution_type} Distribution:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {np.round(sensitivity, 4)}")
    print(f"Specificity: {np.round(specificity, 4)}\n")

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {distribution_type} Distribution")
    plt.show()


def test_distributions(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=43
    )
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=["class"])
    y_test = pd.DataFrame(y_test, columns=["class"])

    nbg_laplace = NaiveBayesCustomDistribution(distribution="laplace")
    feature_stats_laplace = nbg_laplace.train(X_train, y_train["class"])
    draw_custom_curves(feature_stats_laplace, distribution="laplace")
    y_predict_laplace = nbg_laplace.predict(X_test)
    calculate_and_display_results(
        y_test["class"], y_predict_laplace, distribution_type="Laplace"
    )

    nbg_gaussian = NaiveBayesCustomDistribution(distribution="gaussian")
    feature_stats_gaussian = nbg_gaussian.train(X_train, y_train["class"])
    draw_custom_curves(feature_stats_gaussian, distribution="gaussian")
    y_predict_gaussian = nbg_gaussian.predict(X_test)
    calculate_and_display_results(
        y_test["class"], y_predict_gaussian, distribution_type="Gaussian"
    )

    nbg_cauchy = NaiveBayesCustomDistribution(distribution="cauchy")
    feature_stats_cauchy = nbg_cauchy.train(X_train, y_train["class"])
    draw_custom_curves_cauchy(feature_stats_cauchy)
    y_predict_cauchy = nbg_cauchy.predict(X_test)
    calculate_and_display_results(
        y_test["class"], y_predict_cauchy, distribution_type="Cauchy"
    )


def test_crossvalidation(X, y):
    cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=44)

    for distribution in ["laplace", "gaussian", "cauchy"]:
        print(f"Results for {distribution.capitalize()} Distribution:")
        sensitivity_values = []
        specificity_values = []
        accuracy_values = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            nbg = NaiveBayesCustomDistribution(distribution=distribution)
            _ = nbg.train(X_train, y_train["class"])
            y_predict = nbg.predict(X_test)

            (
                conf_matrix,
                accuracy,
                sensitivity,
                specificity,
            ) = calculate_confusion_matrix(y_test["class"], y_predict)

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
    # X = X.drop(columns=["petal width"])

    print("======= DISTRIBUTION COMPARISON TEST =======")
    test_distributions(X, y)
    print("\n========== CROSS-VALIDATION TEST ===========")
    test_crossvalidation(X, y)
