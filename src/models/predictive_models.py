import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from prophet import Prophet
import joblib
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilPricePredictor:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialise le prédicteur de prix du pétrole.
        
        Args:
            model_type: Type de modèle à utiliser ('xgboost', 'random_forest', 'prophet')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price'
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les features pour l'entraînement du modèle.
        
        Args:
            df: DataFrame avec les données historiques
            
        Returns:
            Tuple contenant les features (X) et la target (y)
        """
        # Sélection des features pertinentes
        feature_cols = [
            'volume', 'high', 'low',
            'dollar_index', 'sp500', 'vix', 'gold',
            'price_change', 'price_volatility', 'price_moving_avg',
            'production_change', 'production_moving_avg',
            'dollar_index_change', 'sp500_change', 'vix_change', 'gold_change',
            'dollar_index_moving_avg', 'sp500_moving_avg', 'vix_moving_avg', 'gold_moving_avg'
        ]
        
        # Vérification de la disponibilité des features
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) < len(feature_cols):
            logger.warning(f"Certaines features ne sont pas disponibles: {set(feature_cols) - set(available_features)}")
        
        self.feature_columns = available_features
        X = df[available_features]
        y = df[self.target_column]
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Entraîne le modèle de prédiction des prix.
        
        Args:
            df: DataFrame avec les données historiques
            test_size: Proportion des données à utiliser pour le test
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        X, y = self.prepare_features(df)
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Sélection et entraînement du modèle
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'prophet':
            # Préparation des données pour Prophet
            prophet_df = df.reset_index().rename(columns={'date': 'ds', 'price': 'y'})
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            self.model.fit(prophet_df)
            return self._evaluate_prophet(prophet_df, test_size)
        
        # Entraînement du modèle
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Importance des features (si disponible)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Fait des prédictions sur de nouvelles données.
        
        Args:
            df: DataFrame avec les features nécessaires
            
        Returns:
            Série avec les prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        if self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=len(df))
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(len(df))
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict(X_scaled), index=df.index)
    
    def _evaluate_prophet(self, df: pd.DataFrame, test_size: float) -> Dict:
        """
        Évalue spécifiquement le modèle Prophet.
        
        Args:
            df: DataFrame avec les données
            test_size: Proportion des données de test
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        forecast = self.model.predict(test_df)
        
        metrics = {
            'mse': mean_squared_error(test_df['y'], forecast['yhat']),
            'rmse': np.sqrt(mean_squared_error(test_df['y'], forecast['yhat'])),
            'mae': mean_absolute_error(test_df['y'], forecast['yhat']),
            'r2': r2_score(test_df['y'], forecast['yhat'])
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné."""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'OilPricePredictor':
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(path)
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_columns = model_data['feature_columns']
        return predictor

class ProductionDeclinePredictor:
    def __init__(self):
        """Initialise le prédicteur de déclin de production."""
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les features pour la prédiction du déclin.
        
        Args:
            df: DataFrame avec les données de production
            
        Returns:
            Tuple contenant les features (X) et la target (y)
        """
        # Calcul des features de production
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['cumulative_production'] = df['daily_production'].cumsum()
        df['production_rate'] = df['daily_production'] / df['daily_production'].iloc[0]
        
        # Features pour le modèle
        feature_cols = ['days_since_start', 'cumulative_production']
        X = df[feature_cols]
        y = df['production_rate']
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Entraîne le modèle de prédiction du déclin.
        
        Args:
            df: DataFrame avec les données de production
            test_size: Proportion des données de test
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        X, y = self.prepare_features(df)
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modèle de régression non-linéaire
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Entraînement
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict_decline(self, 
                       initial_production: float,
                       days: int = 365*20) -> pd.DataFrame:
        """
        Prédit le déclin de production sur une période donnée.
        
        Args:
            initial_production: Production initiale (bbl/jour)
            days: Nombre de jours à prédire
            
        Returns:
            DataFrame avec les prédictions de production
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        # Création des features pour la prédiction
        future_dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=days,
            freq='D'
        )
        
        X_pred = pd.DataFrame({
            'days_since_start': range(days),
            'cumulative_production': np.zeros(days)  # Sera mis à jour
        })
        
        # Prédiction itérative
        daily_production = []
        cumulative = 0
        
        for i in range(days):
            X_scaled = self.scaler.transform(X_pred.iloc[[i]])
            decline_rate = self.model.predict(X_scaled)[0]
            daily_prod = initial_production * decline_rate
            daily_production.append(daily_prod)
            cumulative += daily_prod
            if i < days - 1:
                X_pred.iloc[i+1, 1] = cumulative
        
        return pd.DataFrame({
            'date': future_dates,
            'daily_production': daily_production,
            'cumulative_production': np.cumsum(daily_production)
        })
    
    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné."""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'ProductionDeclinePredictor':
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(path)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        return predictor 