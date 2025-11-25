# Projekt BPC-UIM / SUIN –  Predikce výskytu diabetu 
Tento projekt implementuje kompletní pipeline pro zpracování dat, trénování klasifikačního modelu a následné otestování jeho výkonu na nezávislé testovací sadě.
Projekt je vytvořen dle zadání a je připraven k automatickému spuštění a hodnocení.

Název zip souboru: 
projekt_Řehůřek_Jakub.zip

│

├── README.md                      # Dokumentace projektu

├── main.py                        # Hlavní trénovací skript

├── testing.py                     # Skript pro otestování finálního modelu

├── training_final                 # Skript k trénování finálního modelu

├── trained_model_svc_final.pkl    # Natrénovaný finální model

├── scaler_svc_final.pkl           # Scaler 

└── requirements.txt               # Seznam knihoven nutných pro spuštění projektu

**Popis projektu:**

Cílem projektu je vytvořit robustní klasifikační model pro detekci diabetu na základě zdravotních parametrů.

**main.py zahrnuje:**
1. Načtení a předzpracování dat
čištění chybných hodnot (na základě fyziologických limitů), imputace chybějících dat pomocí KNNImputer, převod chybových hodnot zpět na integer typy, kde je to relevantní.

2. Rozdělení datasetu
70 % trénovací sada, 15 % validační sada, 15 % testovací sada

3. Trénování modelu
podpora modelů: Logistic Regression, Random Forest, XGBoost, SVC, možnost škálování pro logreg a SVC, uložení finálního modelu ve formátu .pkl

4. Vyhodnocení
confusion matrix, Matthews correlation coefficient (MCC)

5. Závěrečné testování
pomocí skriptu testing.py, test probíhá na externích datech, které nejsou součástí odevzdaného projektu

**testing.py:**

V dřívějších verzích projektu byl tento skript určen k testování více různých modelů. V aktuální verzi slouží výhradně k testování finálního uloženého modelu, který vznikl v main.py.
Testovací skript: načte trénovaný model a testovací data, provede stejné předzpracování jako při trénování, použije scaler (pokud byl uložen), vypočítá predikce a metriky, zobrazí confusion matrix.

**training_final.py**

Skript slouží k natrénování finální verze modelu. Natrénovaný model je již součástí .zip souboru.
Na rozdíl od hlavního trénovacího skriptu (main.py), který rozděluje data na trénovací, validační a testovací sady, tento skript trénuje model:
na celém dostupném datasetu, bez rozdělení na validační a testovací části, s finální konfigurací hyperparametrů, s plnohodnotným škálováním vstupních dat. 

**requirements.txt**

Tento projekt používá pouze knihovny uvedené v tomto souboru. Ostatní importované knihovny a funkce by měli být součástí Pythonu nebo ostatních skriptů v .zip souboru

**Jak spustit projekt:**

Nainstalujte požadované knihovny: pip install -r requirements.txt a otestujte finální model python testing.py. Před spuštěním nezapomeňte doplnit cestu k testovací sadě do testing.py.

**Autoři:**

Jakub Řehůřek 257040, Andrej Kolář 257008, Jakub Smilowski 257043