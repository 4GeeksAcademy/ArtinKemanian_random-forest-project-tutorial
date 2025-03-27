from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump

datos_train = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/ArtinKemanian_decision-tree-project-tutorial/refs/heads/main/data/processed/datos_limpios_train.csv")
datos_test = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/ArtinKemanian_decision-tree-project-tutorial/refs/heads/main/data/processed/datos_limpios_train.csv")

X_train = datos_train.drop(["Outcome"], axis = 1)
y_train = datos_train["Outcome"]
X_test = datos_test.drop(["Outcome"], axis = 1)
y_test = datos_test["Outcome"]

modelado = RandomForestClassifier(n_estimators = 60, random_state = 42)
modelado.fit(X_train, y_train)

y_pred = modelado.predict(X_test)

accuracy_score(y_test, y_pred)

dump(modelado, open("models/ranfor_classifier", "wb"))