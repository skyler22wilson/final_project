from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
from itertools import product

def find_best_hyperparameters(X, y):
    alpha_values = np.linspace(0.1, 0.99, num=20)
    beta_values = np.linspace(0.1, 0.99, num=20)
    gamma_values = np.linspace(0.1, 0.99, num=20)

    best_alpha = None
    best_beta = None
    best_gamma = None
    best_mse = float('inf')
    
    for alpha, beta, gamma in product(alpha_values, beta_values, gamma_values):
        model = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12)
        fit_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        forecast = fit_model.forecast(steps=len(X))
        
        mse = mean_squared_error(y, forecast)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            
    return best_alpha, best_beta, best_gamma, best_mse




