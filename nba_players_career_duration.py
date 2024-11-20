import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Charger le fichier CSV
data_nba = pd.read_csv('nba-players.csv')

# Afficher les premières lignes pour examiner les données
print(data_nba.head())

# Extraire les colonnes pertinentes
X = data_nba[['gp', 'min']]
Y = data_nba['target_5yrs']

# Initialiser l'arbre de décision
clf = DecisionTreeClassifier(max_depth=7, random_state=42)

# Entraîner l'arbre sur les données
clf.fit(X, Y)

# Visualisation de l'arbre
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No (0)', 'Yes (1)'], filled=True)
plt.show()
