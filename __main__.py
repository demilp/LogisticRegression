from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression
import numpy as np

def run():
    mnist = fetch_mldata('MNIST original')
    X = mnist['data'].astype(float)
    y = mnist['target']
    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = y.reshape(y.shape[0],1)
    y = (y == 5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    np.save('weights', lr.weights)

if __name__ == '__main__':
    run()
