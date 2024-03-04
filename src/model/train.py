import argparse
import os
import wandb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Parser para los argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Load data
wbcd = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log model and evaluation metrics to W&B
with wandb.init(project="Practica_MLOps", name="RandomForest", id=args.IdExecution):
    # Log model
    wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, model_name="RandomForest")

    # Log evaluation metrics
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    wandb.log({"accuracy": accuracy})
