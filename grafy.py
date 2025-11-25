# === Import knihoven === #
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def stats_boxplts_corr_heatmap(file_path:str):
    """
    Funkce načte CSV soubor, vypíše základní statistiky a zobrazi korelační heatmapu.

    Funkce provádí následující kroky:
    - Načtení CSV douboru do DataFrame
    - Zobrazení boxplotů pro každý sloupec
    - Spočítání a zobrazení nulových hodnot pro každý sloupec
    - Vypsání souhrných statistik
    - Vykreslení korelační heatmapy

    Parametry
    ----------
    file_path : str
        Cesta k CSV souboru, který má být načten a analyzován.

    Návratové hodnoty
    -------
        None
    """

    df = pd.read_csv(file_path)
    for i in df.columns:
        data = df[i].dropna()

        mn = data.mean()
        med = data.median()
        stand_dev = data.std()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)


        plt.figure()
        sns.boxplot(y=data, showmeans = True,
                    meanprops={"marker" : "o", "markerfacecolor" : "red", "markeredgecolor" : "black"},
                    boxprops={"facecolor" : "lightblue"},
                    medianprops={"color" : "blue"})

        plt.title(f"{i}")
        plt.xlabel(i)

        stats = (f"Mean = {mn:.2f}\n"
                 f"Median = {med:.2f}\n"
                 f"Std = {stand_dev:.2f}\n"
                 f"Q1 = {q1:.2f}\n"
                 f"Q3 = {q3:.2f}\n")

        plt.text(0.95, 0.85, stats, ha = "center", va = "bottom", fontsize= 9,
                transform=plt.gca().transAxes,
                bbox={"facecolor" : "white", "alpha" : 0.5, "edgecolor" : "gray"})

        plt.show()

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

    return None


def show_plots_and_stats(idx:int):
    """
    Funkce zobrazí krabicové grafy, souhrné statistiky a korelační heatmapu
    pro vybraný soubor.
    Pokud je název souboru neplatný zobrazí se chybová hláška.

    Funkce načte soubor podle zadaného indexu a následně vykreslí:
    - krabicové grafy
    - souhrné statistiky
    - korelační heatmapu

    Parametry
    ----------
    idx : int
        Indexy souboru v seznamu "test_path_file_list".
        Hodnoty od 0 do 6

    Návratové hodnoty
    -------
        None
    """


    test_path_file_list = ["diabetes_data.csv",
                           "test_preprocessed_NAN.csv",
                           "train_preprocessed_NAN.csv",
                           "test_preprocessed_IMPUTE.csv",
                           "train_preprocessed_IMPUTE.csv",
                           "train_preprocessed_ZSCORE.csv",
                           "test_preprocessed_ZSCORE.csv"]

    if idx < len(test_path_file_list):
        stats_boxplts_corr_heatmap(test_path_file_list[idx])
    else:
        print(f"{idx} is invalid input, it must be an integer")

    return None


