from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd



def roc_krivka(model, data):
    '''
    Vypočítá a zobrazí ROC křivku
    :param model: object
        Klasifikační model, který byl vytrénovaný a podporuje metodu 'predict_proba()'
    :param data_testovaci: str
        Cesta k CSV s testovacimi/validačními/trenovacními daty
    Returns
    -------
    roc_auc : float
        Hodnota plochy pod křivkou ROC

    '''

    test_df = pd.read_csv(data)

    x = test_df.drop(columns=['Outcome']) #Oddělí pouze na vstupní příznaky, matice
    y = test_df['Outcome'] #Binární hodnoty cílové proměnné

    y_scores = model.predit_proba(x)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='red', lw=2, label = f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='blue', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

    print(f'ROC AUC: {roc_auc:.4f}')
    return roc_auc
