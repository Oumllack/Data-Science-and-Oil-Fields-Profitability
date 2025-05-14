"""
Module de génération des visualisations pour l'analyse de rentabilité des champs pétroliers.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys

# Ajout du répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.data_collector import OilDataCollector

# Configuration des styles de visualisation
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

class OilVisualisationGenerator:
    def __init__(self, output_dir: str = 'reports/figures'):
        """
        Initialise le générateur de visualisations.
        
        Args:
            output_dir: Répertoire de sortie pour les figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = OilDataCollector()
        
    def plot_historical_oil_prices(self, start_date: str = '2010-01-01', end_date: str = None) -> None:
        """
        Génère un graphique des prix historiques du pétrole.
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
        """
        df = self.collector.get_historical_oil_prices(start_date, end_date)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['price'], label='Prix du pétrole (WTI)')
        plt.title('Évolution des prix du pétrole (WTI)')
        plt.xlabel('Date')
        plt.ylabel('Prix (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'historical_oil_prices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_market_indicators(self, start_date: str = '2010-01-01', end_date: str = None) -> None:
        """
        Génère un graphique des indicateurs de marché.
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
        """
        df = self.collector.get_market_indicators(start_date, end_date)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Indicateurs de marché', fontsize=16)
        
        # Indice du dollar
        axes[0, 0].plot(df['date'], df['dollar_index'], label='Indice du dollar')
        axes[0, 0].set_title('Indice du dollar')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Valeur')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # S&P 500
        axes[0, 1].plot(df['date'], df['sp500'], label='S&P 500')
        axes[0, 1].set_title('S&P 500')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Valeur')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # VIX
        axes[1, 0].plot(df['date'], df['vix'], label='VIX')
        axes[1, 0].set_title('Indice de volatilité (VIX)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Valeur')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Prix de l'or
        axes[1, 1].plot(df['date'], df['gold'], label='Prix de l\'or')
        axes[1, 1].set_title('Prix de l\'or')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Prix (USD)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self, start_date: str = '2010-01-01', end_date: str = None) -> None:
        """
        Génère une matrice de corrélation entre les variables les plus pertinentes.
        
        Args:
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD' (par défaut: aujourd'hui)
        """
        df = self.collector.prepare_training_data(start_date, end_date)
        
        # Sélection des variables les plus pertinentes
        selected_columns = [
            'price',           # Prix du pétrole
            'dollar_index',    # Indice du dollar
            'sp500',          # S&P 500
            'vix',            # Indice de volatilité
            'gold',           # Prix de l'or
            'price_volatility' # Volatilité des prix
        ]
        
        # Calcul de la matrice de corrélation
        corr_matrix = df[selected_columns].corr()
        
        # Création de la figure avec une taille plus grande
        plt.figure(figsize=(12, 10))
        
        # Création de la heatmap avec des annotations plus lisibles
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Masque pour le triangle supérieur
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',  # Format des annotations : 2 décimales
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .8},
            annot_kws={'size': 10}
        )
        
        # Personnalisation des labels
        labels = {
            'price': 'Prix du pétrole',
            'dollar_index': 'Indice du dollar',
            'sp500': 'S&P 500',
            'vix': 'VIX',
            'gold': 'Prix de l\'or',
            'price_volatility': 'Volatilité des prix'
        }
        
        plt.xticks(
            np.arange(len(selected_columns)) + 0.5,
            [labels[col] for col in selected_columns],
            rotation=45,
            ha='right'
        )
        plt.yticks(
            np.arange(len(selected_columns)) + 0.5,
            [labels[col] for col in selected_columns],
            rotation=0
        )
        
        # Ajout du titre et ajustement de la mise en page
        plt.title('Matrice de corrélation des variables clés', pad=20, size=14)
        plt.tight_layout()
        
        # Sauvegarde de la figure avec une meilleure résolution
        plt.savefig(
            self.output_dir / 'correlation_matrix.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        
    def plot_field_production(self, field_name: str, decline_rate: float, initial_production: float, years: int = 20) -> None:
        """
        Génère un graphique de la production d'un champ pétrolier.
        
        Args:
            field_name: Nom du champ
            decline_rate: Taux de déclin annuel
            initial_production: Production initiale (bbl/jour)
            years: Nombre d'années de production
        """
        df = self.collector.get_field_production_data(field_name, decline_rate, initial_production, years)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['daily_production'], label=f'Production du champ {field_name}')
        plt.title(f'Production du champ {field_name}')
        plt.xlabel('Date')
        plt.ylabel('Production (bbl/jour)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'field_production_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_all_visualisations(self) -> None:
        """
        Génère toutes les visualisations.
        """
        print("Génération des visualisations...")
        
        # Prix historiques du pétrole
        print("Génération du graphique des prix historiques du pétrole...")
        self.plot_historical_oil_prices()
        
        # Indicateurs de marché
        print("Génération du graphique des indicateurs de marché...")
        self.plot_market_indicators()
        
        # Matrice de corrélation
        print("Génération de la matrice de corrélation...")
        self.plot_correlation_matrix()
        
        # Production des champs
        print("Génération des graphiques de production des champs...")
        fields = [
            ('Champ A', 0.15, 50000),
            ('Champ B', 0.20, 30000),
            ('Champ C', 0.10, 70000)
        ]
        for field_name, decline_rate, initial_production in fields:
            self.plot_field_production(field_name, decline_rate, initial_production)
        
        print("Génération des visualisations terminée.")

if __name__ == "__main__":
    # Génération de toutes les visualisations
    visualisation_generator = OilVisualisationGenerator()
    visualisation_generator.plot_all_visualisations() 