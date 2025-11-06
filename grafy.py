import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def boxplots(file_path:str):
    """
    Reads csv file, extracs numeric data from each column and generate boxplots
    with statistics - mean, meadian, std, Q1 and Q3.
    :param file_path:(str) path to csv file
    :return: None
    """

    with open(file_path, mode="r") as dta:
        dt = csv.reader(dta)
        hdr = next(dt)
        dt_ar = [row for row in dt]

    dt_ar = np.array(dt_ar, dtype=object)

    for i in range(len(hdr)):
        col_data = dt_ar[:, i]


        data = []
        for val in col_data:
            try:
                if val != "":
                    data.append(float(val))
            except ValueError:
                continue

        data = np.array(data, dtype=float)


        if data.size == 0:
            continue


        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        q1, q3 = np.percentile(data, [25, 75])


        sns.boxplot(data=data, showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
                    boxprops={"facecolor": "lightblue"},
                    medianprops={"color": "blue"})

        plt.title(f"{hdr[i]} test")
        plt.xlabel(hdr[i])


        text = f"Mean = {mean_val:.2f}\nMedian = {median_val:.2f}\nStd = {std_val:.2f}\nQ1 = {q1:.2f}\nQ3 = {q3:.2f}"
        plt.text(0.5, max(data), text, ha='center', va='bottom', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

        plt.show()


def stats_and_corr_heatmap(file_path:str):
    """
    Reads csv file into pandas dataframe,
    displays statistics count, means, std, min, max and percentiles for all columns,
    counts the number of zero values per column, then displays correlation heatmap.
    :param file_path: (str) Path to csv file
    :return: None
    """

    df = pd.read_csv(file_path)

    head = list(df.columns)
    for i in head:
        c = 0
        for j in df[i]:
            if j == 0:
                c += 1

        print(i,c)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", None)

    print(df.describe())


    plt.figure(figsize=(12,12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm').set_title('Correlation Heatmap')
    plt.show()



def show_plots_and_stats(idx:int):
    """
    The function displays boxplots, summary statistics, and a correlation heatmap for the selected file,
    if the input is invalid, an error message will be shown.
    :param idx: (int) Index of file path you want to use, integer - 0, 1 or 2
    :return: None
    """


    test_path_file_list = ["diabetes_data.csv", "test_preprocessed_NAN.csv", "train_preprocessed_NAN.csv", "test_preprocessed_IMPUTE.csv","train_preprocessed_IMPUTE.csv",
                           "train_preprocessed_ZSCORE.csv","test_preprocessed_ZSCORE.csv"]

    if idx == 0:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 1:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 2:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 3:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 4:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 5:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])
    elif idx == 6:
        boxplots(test_path_file_list[idx])
        stats_and_corr_heatmap(test_path_file_list[idx])

    else:
        print(f"{idx} is invalid input, it must be an integer")

show_plots_and_stats(6)