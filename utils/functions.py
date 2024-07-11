import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import bootstrap
from sklearn.metrics import confusion_matrix


def histogram(dataframe: pd.DataFrame, feature_name: str) -> None:
    """
    Plots a histogram for a specified feature in the given DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        feature_name (str): The name of the feature (column) for which to plot the histogram.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(data=dataframe, x=feature_name, zorder=2)
    name = " ".join(word.capitalize() for word in feature_name.split())
    plt.title(f"{name} Distribution")
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.tick_params(axis="both", labelsize=9)
    plt.tick_params(axis="both", length=0)
    plt.grid(alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True)


def relationship(dataframe: pd.DataFrame, feature_name: str) -> None:
    """
    Plot a histogram and boxplot with correlation coefficient and p-value between 'quality' and another feature.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        feature_name (str): The name of the feature to plot against "quality".

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=dataframe, x=feature_name, ax=axs[0], zorder=2)
    name = " ".join(word.capitalize() for word in feature_name.split())
    axs[0].set_title(f"{name} Distribution")
    axs[0].set_xlabel(name)
    axs[0].set_ylabel("Frequency")
    axs[0].tick_params(axis="both", labelsize=9)
    axs[0].grid(alpha=0.2, zorder=0)
    sns.despine(ax=axs[0], left=True, bottom=True)
    axs[0].tick_params(axis="both", length=0)
    sns.boxplot(data=dataframe, y=feature_name, x="quality", ax=axs[1], zorder=2)
    axs[1].set_title(f"Distribution of {name} by Quality Levels")
    axs[1].set_ylabel(name)
    axs[1].set_xlabel("Quality")
    axs[1].tick_params(axis="both", labelsize=9)
    axs[1].grid(alpha=0.2, axis="y", zorder=0)
    sns.despine(ax=axs[1], left=True, bottom=True)
    axs[1].tick_params(axis="both", length=0)
    plt.tight_layout()
    plt.show()
    corr_coef, p_value = spearmanr(dataframe[feature_name], dataframe["quality"])
    print(f"Spearman correlation coefficient: {corr_coef:.2f}")
    print(f"p-value: {p_value:.2f}")


def median_confidence_intervals(dataframe: pd.DataFrame, feature_name: str) -> None:
    """
    Calculate and print the 95% confidence intervals for the median of a specified feature,
    grouped by 'quality' in the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    feature_name (str): The name of the feature/column for which to calculate the median confidence intervals.

    Returns:
    None: This function prints the confidence intervals for each quality group.
    """
    confidence_intervals = dataframe.groupby("quality")[feature_name].apply(
        lambda x: bootstrap(
            (x.values,),
            np.median,
            confidence_level=0.95,
            n_resamples=1000,
            method="percentile",
        ).confidence_interval
    )
    for quality, ci in confidence_intervals.items():
        print(f"Quality {quality}: [{ci.low:.2f}, {ci.high:.2f}]")


def quality_count_plot(dataframe: pd.DataFrame) -> None:
    """
    Plots a count plot for the 'quality' feature in the given DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=dataframe, x="quality", zorder=2)
    plt.title("Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Frequency")
    plt.tick_params(axis="both", labelsize=9)
    plt.grid(alpha=0.2, axis="y", zorder=0)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis="both", length=0)


def correlation_heatmap(dataframe: pd.DataFrame, target_feature_name: str) -> None:
    """
    Generate a heatmap to visualize the Spearman correlation matrix, along with the p-values.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing the features and target variable.
        target_feature_name (str): The name of the target feature to be excluded from the correlation matrix.

    Returns:
        None
    """
    features_to_correlate = dataframe.drop(columns=[target_feature_name])
    corr_matrix, p_matrix = spearmanr(features_to_correlate)
    mask = np.triu(np.ones_like(corr_matrix), k=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        mask=mask,
        cbar=False,
    )
    for i in range(len(features_to_correlate.columns)):
        for j in range(len(features_to_correlate.columns)):
            if not mask[i, j]:
                p_value = p_matrix[i, j]
                if not np.isnan(p_value):
                    plt.text(
                        j + 0.518,
                        i + 0.75,
                        f"({p_value:.2f})",
                        ha="center",
                        va="center",
                        fontsize=7,
                        alpha=0.7,
                    )
    plt.xticks(
        ticks=np.arange(len(features_to_correlate.columns)) + 0.5,
        labels=features_to_correlate.columns,
        rotation=90,
    )
    plt.yticks(
        ticks=np.arange(len(features_to_correlate.columns)) + 0.5,
        labels=features_to_correlate.columns,
        rotation=0,
    )
    plt.title("Correlation Heatmap With p-values Between All Predictors", loc="left")


def plot_confusion(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plots the confusion matrices for the training and test datasets.

    Args:
        y_train (pd.Series): Actual training data.
        y_train_pred (np.ndarray): Predicted values for training set.
        y_test (pd.Series): Actual test data.
        y_test_pred (np.ndarray): Predicted values for testing set.

    Returns:
        None
    """
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    labels = sorted(y_train.unique())
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        conf_matrix_train,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.7,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Training Set Confusion Matrix")
    plt.subplot(1, 2, 2)
    sns.heatmap(
        conf_matrix_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.7,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()


def plot_residuals(train_residuals: pd.Series, test_residuals: pd.Series) -> None:
    """
    Plots the residuals for the training and test datasets.

    Args:
        train_residuals (pd.Series): Residuals from the training set.
        test_residuals (pd.Series): Residuals from the test set.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=train_residuals, ax=ax[0], zorder=2)
    ax[0].set_xlabel("Residual")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Residuals for Training Set")
    ax[0].tick_params(axis="both", labelsize=9)
    ax[0].tick_params(axis="both", length=0)
    ax[0].grid(axis="y", alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True, ax=ax[0])
    sns.countplot(x=test_residuals, ax=ax[1], zorder=2)
    ax[1].set_xlabel("Residual")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Residuals for Test Set")
    ax[1].tick_params(axis="both", labelsize=9)
    ax[1].tick_params(axis="both", length=0)
    ax[1].grid(axis="y", alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True, ax=ax[1])
    plt.tight_layout()


def plot_leverage_cooksd(
    leverage: np.ndarray, cooks_d: np.ndarray, p: int, n: int
) -> None:
    """
    Plots the leverage and Cook's Distance for each observation.

    Args:
        leverage (np.ndarray): Leverage values for each observation.
        cooks_d (np.ndarray): Cook's Distance values for each observation.
        p (int): Number of predictors in the model.
        n (int): Number of observations.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x=np.arange(len(leverage)), y=leverage, alpha=0.7, ax=ax[0])
    ax[0].axhline(y=2 * (p + 1) / n, color="r", linestyle="--")
    ax[0].set_xlabel("Observation Index")
    ax[0].set_ylabel("Leverage")
    ax[0].set_title("Leverage of Each Observation")
    ax[0].tick_params(axis="both", labelsize=9)
    ax[0].tick_params(axis="both", length=0)
    ax[0].grid(alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True, ax=ax[0])
    sns.scatterplot(x=np.arange(len(cooks_d)), y=cooks_d, alpha=0.7, ax=ax[1])
    ax[1].axhline(y=4 / n, color="r", linestyle="--")
    ax[1].set_xlabel("Observation Index")
    ax[1].set_ylabel("Cook's Distance")
    ax[1].set_title("Cook's Distance for Each Observation")
    ax[1].tick_params(axis="both", labelsize=9)
    ax[1].tick_params(axis="both", length=0)
    ax[1].grid(alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True, ax=ax[1])
    plt.tight_layout()


def coefficients_with_intervals(
    features: pd.Index, mean_coeffs: np.ndarray, error_bars: np.ndarray
) -> None:
    """
    Plots the regression coefficient estimates with 95% confidence intervals.

    Args:
        features (pd.Index): The feature names.
        mean_coeffs (np.ndarray): The mean coefficient estimates.
        error_bars (np.ndarray): The error bars (confidence intervals) for the coefficients.

    Returns:
        None
    """
    plt.figure(figsize=(9, 5))
    plt.barh(
        features, mean_coeffs, xerr=error_bars, align="center", capsize=9, zorder=2
    )
    plt.xlabel("Coefficient Value")
    plt.ylabel("Features")
    plt.title("Regression Coefficient Estimates with 95% Confidence Intervals")
    plt.tick_params(axis="both", labelsize=9)
    plt.tick_params(axis="both", length=0)
    plt.grid(alpha=0.2, zorder=0)
    sns.despine(left=True, bottom=True)
