import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class NaiveBayesCustomDistribution:
    def __init__(self, distribution="gaussian"):
        self.unique_classes = None
        self.prior_probs = None
        self.feature_stats = {}
        self.distribution = distribution

    def calculate_likelihood(self, feature_name, feature_val, label):
        class_data = self.feature_stats[label]
        params = class_data[feature_name]

        if self.distribution == "gaussian":
            return self.calculate_likelihood_gaussian(
                feature_val, params["mean"], params["b"]
            )
        elif self.distribution == "laplace":
            return self.calculate_likelihood_laplace(
                feature_val, params["mean"], params["b"]
            )
        elif self.distribution == "cauchy":
            return self.calculate_likelihood_cauchy(
                feature_val, params["mean"], params["b"]
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def calculate_likelihood_gaussian(self, feature_val, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
            -((feature_val - mean) ** 2 / (2 * std ** 2))
        )

    def calculate_likelihood_laplace(self, feature_val, mean, b):
        return (1 / (2 * b)) * np.exp(-np.abs(feature_val - mean) / b)

    def calculate_likelihood_cauchy(self, feature_val, location, scale):
        return 1 / (np.pi * scale * (1 + ((feature_val - location) / scale) ** 2))

    def train(self, X, y):
        self.unique_classes, class_counts = np.unique(y, return_counts=True)
        self.prior_probs = class_counts / len(y)

        for target in self.unique_classes:
            class_mask = y == target
            class_data = X.loc[class_mask.values.flatten()]
            feature_stats_cls = {}

            for feature in X.columns:
                if self.distribution == "gaussian":
                    mean, b = class_data[feature].mean(), class_data[feature].std()
                elif self.distribution == "laplace":
                    mean, b = class_data[feature].mean(), np.mean(
                        np.abs(class_data[feature] - class_data[feature].mean())
                    )
                elif self.distribution == "cauchy":
                    mean, sorted_vals = class_data[feature].median(), np.sort(
                        np.abs(class_data[feature] - class_data[feature].median())
                    )
                    b = sorted_vals[int(len(sorted_vals) * 0.75)]
                else:
                    raise ValueError(f"Unsupported distribution: {self.distribution}")

                feature_stats_cls[feature] = {"mean": mean, "b": b}

            self.feature_stats[target] = feature_stats_cls
        return self.feature_stats

    def predict(self, X_test):
        y_pred = []

        for _, row in X_test.iterrows():
            class_scores = np.zeros(len(self.unique_classes))

            for class_idx, class_label in enumerate(self.unique_classes):
                combined_feature_prob = 1.0

                for feature in X_test.columns:
                    feature_val_prob = self.calculate_likelihood(
                        feature, row[feature], class_label
                    )
                    combined_feature_prob *= feature_val_prob

                class_scores[class_idx] = (
                    self.prior_probs[class_idx] * combined_feature_prob
                )

            predicted_class = self.unique_classes[np.argmax(class_scores)]
            y_pred.append(predicted_class)

        return np.array(y_pred)


def laplace_curve(x, mean, b):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mean) / b)


def gaussian_curve(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mean) ** 2) / (2 * std ** 2)
    )


def cauchy_curve(x, location, scale):
    return 1 / (np.pi * scale * (1 + ((x - location) / scale) ** 2))


def draw_custom_curves(feature_stats, distribution="gaussian"):
    for class_label, features in feature_stats.items():
        plt.suptitle(f"Class: {class_label}")
        for feature, params in features.items():
            x = np.linspace(
                params["mean"] - 3 * params["b"], params["mean"] + 3 * params["b"], 100
            )
            if distribution == "gaussian":
                y = gaussian_curve(x, params["mean"], params["b"])
            elif distribution == "laplace":
                y = laplace_curve(x, params["mean"], params["b"])
            elif distribution == "cauchy":
                y = cauchy_curve(x, params["mean"], params["b"])
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

            plt.plot(x, y, label=f"{feature}")
        plt.legend()
        plt.show()


def calculate_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    accuracy = np.sum(TP) / np.sum(conf_matrix)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return conf_matrix, accuracy, sensitivity, specificity


if __name__ == "__main__":
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features, columns=iris.data.feature_names)
    y = pd.DataFrame(iris.data.targets, columns=["class"])
    # X = X.drop(columns=["petal length"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=["class"])
    y_test = pd.DataFrame(y_test, columns=["class"])

    nbg = NaiveBayesCustomDistribution(distribution="gaussian")
    feature_stats_gaussian = nbg.train(X_train, y_train["class"])
    draw_custom_curves(feature_stats_gaussian, distribution="gaussian")
    y_predict = nbg.predict(X_test)

    conf_matrix, accuracy, sensitivity, specificity = calculate_confusion_matrix(
        y_test["class"], y_predict
    )

    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity: {np.round(sensitivity, 2)}")
    print(f"Specificity: {np.round(specificity, 2)}\n")

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_test["class"]),
        yticklabels=np.unique(y_test["class"]),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Cauchy Distribution")
    plt.show()
