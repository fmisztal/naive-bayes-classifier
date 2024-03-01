import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class NaiveBayesClassifier:
    def __init__(self):
        self.unique_classes = None
        self.class_probs = None
        self.feature_probs = {}

    def train(self, X, y):
        self.unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = class_counts / len(y)

        for target in self.unique_classes:
            class_mask = y == target
            class_data = X.loc[class_mask.values.flatten()]
            feature_probs_cls = {}

            for feature in X.columns:
                unique_values, value_counts = np.unique(
                    class_data[feature], return_counts=True
                )
                feature_probs_cls[feature] = dict(
                    zip(unique_values, value_counts / len(class_data))
                )

            self.feature_probs[target] = feature_probs_cls

    def predict(self, dataframe):
        predicted_classes = []

        for _, row in dataframe.iterrows():
            class_scores = np.zeros(len(self.unique_classes))

            for class_idx, class_label in enumerate(self.unique_classes):
                combined_feature_prob = 1.0

                for feature in dataframe.columns:
                    feature_val_prob = (
                        self.feature_probs.get(class_label, {})
                        .get(feature, {})
                        .get(row[feature], 1e-10)
                    )
                    combined_feature_prob *= feature_val_prob

                class_scores[class_idx] = (
                    self.class_probs[class_idx] * combined_feature_prob
                )

            predicted_class = self.unique_classes[np.argmax(class_scores)]
            predicted_classes.append(predicted_class)

        return predicted_classes


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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    df_X_train = pd.DataFrame(X_train, columns=X.columns)
    df_X_test = pd.DataFrame(X_test, columns=X.columns)
    df_y_train = pd.DataFrame(y_train, columns=["class"])
    df_y_test = pd.DataFrame(y_test, columns=["class"])

    nbc = NaiveBayesClassifier()
    nbc.train(df_X_train, df_y_train)
    y_predict = nbc.predict(df_X_test)

    conf_matrix = confusion_matrix(df_y_test["class"], y_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(df_y_test["class"]),
        yticklabels=np.unique(df_y_test["class"]),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    accuracy = np.sum(TP) / np.sum(conf_matrix)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
