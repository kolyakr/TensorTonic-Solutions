import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

# def bce(y_true, y_pred):
#     eps = 1e-15
#     y_pred = np.clip(y_pred, eps, 1 - eps)
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def grads(y_true, y_pred, X, N):
    error = y_pred - y_true 
    
    dL_dw = (X.T @ error) / N
    dL_db = np.mean(error) 

    return dL_dw, dL_db

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X_train = np.array(X)
    y_train = np.array(y).reshape(-1, 1) 
    
    N, D = X_train.shape

    w = np.random.randn(D, 1) * 0.01
    b = 0.0
    
    for i in range(steps):
        y_pred = _sigmoid(X_train @ w + b)
        
        dw, db = grads(y_train, y_pred, X_train, N)
        
        w -= lr * dw
        b -= lr * db
    
    return w.flatten(), float(b)