from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def chybova_krivka(model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=5, scoring='accuracy', train_sizes = np.linspace(0.1, 1.0, 10),
        n_jobs = 1, random_state = 42
    )

    train_error = 1 - np.mean(train_scores, axis=1)
    test_error = 1 - np.mean(test_scores, axis=1)
    #Vykresleni grafu :)
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, train_error, 'k--', label='training data')
    plt.plot(train_sizes, test_error, 'r-', label='testing data')
    plt.title('Overfiting check')
    plt.xlabel('Pocet vzroku')
    plt.ylabel('Chyba')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()