
'''
    @author Munawar Hasan <munawar.hasan@nist.gov>
'''

from qml import Model

import pandas as pd
import numpy as np

data_dir = "data/"
tr_file = data_dir +"tr_probabilities.csv"
te_file = data_dir +"te_probabilities.csv"

unitary_matrix = [
    [
        -0.33500492 - 0.71478153j, 0.32202088 - 0.31921061j,
        -0.19005727 - 0.26556473j, 0.25141268 - 0.03756969j
    ],
    [
        0.06510655 - 0.12600615j, 0.06592668 - 0.61643856j,
        0.35661372 + 0.21461067j, -0.60806694 + 0.229269j
    ],
    [
        0.36498115 - 0.44204005j, -0.56451608 - 0.01413113j,
        -0.47660056 + 0.27633166j, -0.15701817 - 0.15604254j
    ],
    [
        -0.0325527 - 0.16453671j, -0.29302758 + 0.07294554j,
        0.51825632 - 0.384741j, -0.15690396 - 0.6629085j
    ]
]


def test_qml(csv_filename):
    df = pd.read_csv(csv_filename)
    print(df.head())
    model_1_proba = df['model_1_proba'].tolist()
    model_2_proba = df['model_2_proba'].tolist()

    labels = df['labels'].tolist()
    labels = list(map(int, labels))
    print(model_1_proba[0], model_2_proba[0], labels[0])

    X = np.zeros([len(model_1_proba), 2], dtype='float32')

    for index in range(len(model_1_proba)):
        X[index, :] = [model_1_proba[index], model_2_proba[index]]
    print("Dataset Shape: ", X.shape)

    model = Model(num_of_qubits=2, U=unitary_matrix)

    y_pred = model.compute(x=X, pred=True, pred_index=None)
    dd = model.get_metrics(y_true=labels, y_pred=y_pred)
    return dd


def test_stub1():
    print("<<evaluating qml on train dataset>> ....")
    train_metrics = test_qml(tr_file)
    print("train metrics: ", train_metrics)

    print("<<evaluating qml on test dataset>> ....")
    test_metrics = test_qml(te_file)
    print("test metrics: ", test_metrics)


if __name__ == '__main__':
    print("qml demo .....")
    test_stub1()
