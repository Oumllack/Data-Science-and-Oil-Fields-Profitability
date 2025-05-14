"""
Module de collecte des données pour l'analyse de rentabilité des champs pétroliers.
"""

import os
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional, Tuple

# Chargement des variables d'environnement
load_dotenv()

class OilDataCollector:
    def __init__(self):
        """Initialise le collecteur de données."""
        pass
        
    def get_historical_oil_prices(self, 
                                start_date: str = '2010-01-01',
                                end_date: str = None) -> pd.DataFrame:
        """
        Récupère les prix historiques du pétrole brut (WTI) depuis Yahoo Finance.
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
            
        Returns:
            DataFrame avec les prix historiques
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Récupération des données WTI
        wti = yf.download('CL=F', start=start_date, end=end_date)
        
        # Nettoyage et préparation des données
        df = pd.DataFrame(index=wti.index)
        df['date'] = df.index
        df['price'] = wti['Close'].values  # Utilisation de .values pour obtenir un tableau 1D
        df['volume'] = wti['Volume'].values
        df['high'] = wti['High'].values
        df['low'] = wti['Low'].values
        
        # Ajout de features temporelles
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        return df
    
    def get_market_indicators(self,
                            start_date: str = '2010-01-01',
                            end_date: str = None) -> pd.DataFrame:
        """
        Récupère les indicateurs de marché pertinents (USD, indices boursiers, etc.).
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
            
        Returns:
            DataFrame avec les indicateurs de marché
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Récupération des données pour différents indicateurs
        indicators = {
            'DX-Y.NYB': 'dollar_index',  # Indice du dollar
            '^GSPC': 'sp500',            # S&P 500
            '^VIX': 'vix',               # Indice de volatilité
            'GC=F': 'gold'               # Prix de l'or
        }
        
        dfs = []
        for ticker, name in indicators.items():
            data = yf.download(ticker, start=start_date, end=end_date)
            df = pd.DataFrame(index=data.index)
            df['date'] = df.index
            df[name] = data['Close'].values  # Utilisation de .values pour obtenir un tableau 1D
            dfs.append(df)
        
        # Fusion des DataFrames
        market_data = dfs[0]
        for df in dfs[1:]:
            market_data = market_data.merge(df, on='date', how='outer')
            
        return market_data
    
    def prepare_training_data(self,
                            start_date: str = '2010-01-01',
                            end_date: str = None) -> pd.DataFrame:
        """
        Prépare un dataset complet pour l'entraînement des modèles.
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
            
        Returns:
            DataFrame avec toutes les features nécessaires pour l'entraînement
        """
        # Récupération des différentes sources de données
        prices_df = self.get_historical_oil_prices(start_date, end_date)
        market_df = self.get_market_indicators(start_date, end_date)
        
        # Fusion des données
        df = prices_df.merge(market_df, on='date', how='outer')
        
        # Nettoyage des données
        df = df.fillna(method='ffill')  # Forward fill pour les valeurs manquantes
        
        # Ajout de features techniques
        df['price_change'] = df['price'].pct_change()
        df['price_volatility'] = df['price'].rolling(window=20).std()
        df['price_moving_avg'] = df['price'].rolling(window=20).mean()
        
        # Features de marché
        for col in ['dollar_index', 'sp500', 'vix', 'gold']:
            df[f'{col}_change'] = df[col].pct_change()
            df[f'{col}_moving_avg'] = df[col].rolling(window=20).mean()
        
        # Suppression des lignes avec des valeurs manquantes
        df = df.dropna()
        
        return df
    
    def get_field_production_data(self, 
                                field_name: str,
                                decline_rate: float,
                                initial_production: float,
                                years: int = 20) -> pd.DataFrame:
        """
        Génère des données de production synthétiques pour un champ pétrolier.
        
        Args:
            field_name: Nom du champ
            decline_rate: Taux de déclin annuel
            initial_production: Production initiale (bbl/jour)
            years: Nombre d'années de production
            
        Returns:
            DataFrame avec les données de production du champ
        """
        dates = pd.date_range(start='2024-01-01', periods=years*365, freq='D')
        daily_production = []
        
        for i in range(len(dates)):
            # Calcul de la production journalière avec déclin exponentiel
            daily_prod = initial_production * (1 - decline_rate) ** (i/365)
            # Ajout de bruit aléatoire pour plus de réalisme
            daily_prod *= np.random.normal(1, 0.05)  # 5% de variation
            daily_production.append(max(0, daily_prod))  # Production non négative
        
        df = pd.DataFrame({
            'date': dates,
            'field_name': field_name,
            'daily_production': daily_production
        })
        
        # Agrégation mensuelle
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        monthly_prod = df.groupby(['year', 'month'])['daily_production'].mean().reset_index()
        monthly_prod['date'] = pd.to_datetime(monthly_prod[['year', 'month']].assign(day=1))
        
        return monthly_prod[['date', 'daily_production']]

if __name__ == "__main__":
    # Exemple d'utilisation
    collector = OilDataCollector()
    
    # Récupération des prix du pétrole
    oil_prices = collector.get_historical_oil_prices()
    print("\nPrix du pétrole:")
    print(oil_prices.head())
    
    # Récupération des indicateurs de marché
    market_data = collector.get_market_indicators()
    print("\nIndicateurs de marché:")
    print(market_data.head())
    
    # Génération de données de production
    production_data = collector.get_field_production_data(
        field_name='Sample Field',
        decline_rate=0.15,
        initial_production=50000,
        years=5
    )
    print("\nDonnées de production:")
    print(production_data.head()) 