"""
modeling.py — builders sklearn-compatibles para todos los modelos
que vamos a comparar.

Modelos:
  1. DummyRegressor (baseline obligatorio)
  2. Ridge          (L2 fuerte — clave con p>>n)
  3. Lasso          (L1)
  4. ElasticNet     (L1+L2)
  5. SVR            (RBF)
  6. RandomForest
  7. XGBoost        (si esta instalado)
  8. MLPRegressor   (sklearn, opcional, NO prioridad)
"""
import warnings
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import RANDOM_SEED

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _scaled(est):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   est),
    ])


def _unscaled(est):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   est),
    ])


def build_dummy():
    return _unscaled(DummyRegressor(strategy='mean'))


def build_ridge(alpha: float = 10.0):
    return _scaled(Ridge(alpha=alpha, random_state=RANDOM_SEED))


def build_lasso(alpha: float = 1.0):
    return _scaled(Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_SEED))


def build_elasticnet(alpha: float = 1.0, l1_ratio: float = 0.5):
    return _scaled(ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                              max_iter=20000, random_state=RANDOM_SEED))


def build_svr(C: float = 10.0, epsilon: float = 1.0, gamma='scale'):
    return _scaled(SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))


def build_rf(n_estimators: int = 200, max_depth=None, min_samples_leaf: int = 2):
    return _unscaled(RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED, n_jobs=-1,
    ))


def build_xgb(n_estimators: int = 200, max_depth: int = 3,
              learning_rate: float = 0.05, subsample: float = 0.8,
              colsample_bytree: float = 0.8, reg_lambda: float = 1.0):
    if not XGBOOST_AVAILABLE:
        return None
    return _unscaled(xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
    ))


def build_mlp(hidden=(32,), alpha: float = 0.01, max_iter: int = 2000):
    """MLPRegressor de sklearn. Con n=8 sera inestable: usar con precaucion."""
    return _scaled(MLPRegressor(
        hidden_layer_sizes=hidden, alpha=alpha,
        max_iter=max_iter, random_state=RANDOM_SEED,
        early_stopping=False,  # con 8 puntos no hay sentido validacion interna
    ))


def all_baseline_models() -> dict:
    """Devuelve dict {nombre: pipeline} con todos los baselines disponibles."""
    models = {
        'DummyRegressor': build_dummy(),
        'Ridge':          build_ridge(),
        'Lasso':          build_lasso(),
        'ElasticNet':     build_elasticnet(),
        'SVR':            build_svr(),
        'RandomForest':   build_rf(),
    }
    xgb_pipe = build_xgb()
    if xgb_pipe is not None:
        models['XGBoost'] = xgb_pipe
    else:
        warnings.warn("[MODEL] XGBoost no instalado, se omite.")
    models['MLP'] = build_mlp()  # sklearn, siempre disponible
    return models
