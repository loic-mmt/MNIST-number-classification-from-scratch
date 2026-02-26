# MNIST Classification: notebook et version C++

Ce dossier contient deux implementations du meme objectif: entrainer un reseau de neurones sur MNIST et produire un fichier de soumission, sans librairie de machine learning.

## 1) `image_classif.ipynb`

Le notebook est une version Python/Numpy, organisee en étapes:

1. Chargement des donnees:
   - lit `train_mnist.csv` et `test_mnist.csv` avec pandas.
   - affiche les shapes et un apercu des colonnes.

2. Exploration visuelle:
   - detecte les colonnes pixels (`1x1` ... `28x28`),
   - reconstruit des images `28x28`,
   - affiche une image unique puis une grille aleatoire.

3. Entrainement d'un MLP (multiclasse):
   - architecture: entree 784 -> couche cachée sigmoid -> sortie 10 classes (softmax),
   - one-hot sur les labels,
   - mini-batch gradient descent,
   - dropout (inverted dropout) sur la couche cachee,
   - suivi de l'accuracy train par epoch,
   - sauvegarde des meilleurs poids/biais (`best`).

4. Prediction et soumission:
   - applique le modele sauvegardé sur le test,
   - crée `submission.csv` avec les colonnes `id,label`.

## 2) `mnist.cpp`

Le fichier C++ (Armadillo) est la transposition du pipeline du notebook:

1. Fonctions mathematiques:
   - `sigmoid`, `sigmoid_deriv`, `softmax`, `one_hot`, `dropout_mask`.

2. Parsing CSV:
   - `CSVRow` lit les lignes CSV rapidement,
   - `parse_header_columns` identifie colonnes `label`, `id` et pixels.

3. Chargement donnees:
   - `load_mnist_data` charge train (`X` + `y`),
   - `load_mnist_test_data` charge test (`X` + `ids`),
   - normalisation des pixels en `[0,1]`.

4. Entrainement:
   - `train_nn` entraine un MLP 784 -> hidden -> 10,
   - softmax + cross-entropy,
   - dropout sur la couche cachee,
   - sauvegarde dans `NNModel` des meilleurs poids/biais selon l'accuracy.

5. Inference et export:
   - `predict_proba` calcule les probabilités,
   - `predict` retourne la classe argmax,
   - `write_submission_csv` écrit le fichier de soumission.

6. `main`:
   - charge train, affiche des checks,
   - entraine le modele,
   - charge test, predit les labels,
   - écrit un fichier de soumission (actuellement appelé `submission_cpp.csv`).

## 3) Relation entre les deux fichiers

- `image_classif.ipynb`: ideation/experimentation rapide en Python.
- `mnist.cpp`: version executable C++ plus structurée pour reproduction et performance.

Les deux suivent la meme logique globale: extraction pixels -> entrainement MLP -> prediction test -> soumission `id,label`.
