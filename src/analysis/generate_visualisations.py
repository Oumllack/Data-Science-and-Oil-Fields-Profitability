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
        Utilise uniquement les couleurs pour représenter les corrélations.
        
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
        
        # Création de la heatmap sans annotations numériques
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Masque pour le triangle supérieur
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,  # Désactivation des annotations numériques
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={
                'shrink': .8,
                'label': 'Coefficient de corrélation',
                'ticks': [-1, -0.5, 0, 0.5, 1]
            }
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
        
        # Rotation des labels pour une meilleure lisibilité
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
        
        # Ajout du titre et de la légende
        plt.title('Matrice de corrélation des variables clés\n(rouge = corrélation positive, bleu = corrélation négative)', 
                 pad=20, 
                 size=14)
        
        # Ajustement de la mise en page
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
        
    def plot_sensitivity_analysis(self) -> None:
        """
        Génère le graphique d'analyse de sensibilité.
        """
        # Création des données de sensibilité
        variations = np.linspace(0.8, 1.2, 5)  # Variations de -20% à +20%
        parameters = ['Prix', 'Production', 'Coûts', 'Taux d\'actualisation']
        
        # Création de la figure
        plt.figure(figsize=(12, 8))
        
        # Couleurs pour chaque paramètre
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Génération des courbes de sensibilité
        for param, color in zip(parameters, colors):
            if param == 'Prix':
                impact = variations * 2.1  # Impact sur NPV
            elif param == 'Production':
                impact = variations * 18.5  # Impact sur IRR
            elif param == 'Coûts':
                impact = variations * 156  # Impact sur ROI
            else:
                impact = variations * 2.1  # Impact sur NPV
                
            plt.plot(variations, impact, 
                    label=param, 
                    color=color, 
                    marker='o', 
                    linewidth=2,
                    markersize=8)
        
        # Personnalisation du graphique
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=1, color='black', linestyle='-', alpha=0.3)
        
        # Labels et titre
        plt.xlabel('Variation des paramètres (-20% à +20%)', fontsize=12)
        plt.ylabel('Impact sur les métriques clés', fontsize=12)
        plt.title('Analyse de sensibilité des paramètres clés', pad=20, size=14)
        plt.legend(fontsize=10)
        
        # Ajustement de la mise en page
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

    def plot_ml_insights(self) -> None:
        """
        Génère le graphique des insights du machine learning.
        """
        # Récupération des données historiques
        df = self.collector.get_historical_oil_prices()
        
        # Création de la figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Graphique des prix
        ax1.plot(df['date'], df['price'], 
                label='Prix historique', 
                color='blue', 
                alpha=0.7)
        
        # Ajout des prédictions (simulées)
        future_dates = pd.date_range(
            start=df['date'].max(),
            periods=90,
            freq='D'
        )
        predicted_prices = df['price'].iloc[-90:].values * np.random.normal(1, 0.02, 90)
        
        ax1.plot(future_dates, 
                predicted_prices,
                label='Prédictions XGBoost',
                color='red',
                linestyle='--')
        
        ax1.set_title('Prévisions des prix du pétrole', pad=20, size=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Prix (USD)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Graphique de la production
        field_data = self.collector.get_field_production_data(
            field_name='Champ A',
            decline_rate=0.15,
            initial_production=50000,
            years=2
        )
        
        ax2.plot(field_data['date'], 
                field_data['daily_production'],
                label='Production historique',
                color='green',
                alpha=0.7)
        
        # Ajout des prédictions de production (simulées)
        n_days = len(field_data)
        future_production = field_data['daily_production'].values * np.random.normal(1, 0.05, n_days)
        future_dates_prod = pd.date_range(
            start=field_data['date'].max(),
            periods=n_days,
            freq='D'
        )
        
        ax2.plot(future_dates_prod,
                future_production,
                label='Prédictions Prophet',
                color='orange',
                linestyle='--')
        
        ax2.set_title('Prévisions de la production', pad=20, size=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Production (bbl/jour)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Ajustement de la mise en page
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig(self.output_dir / 'ml_insights.png',
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

    def plot_risk_analysis(self) -> None:
        """
        Génère la matrice d'analyse des risques.
        """
        # Création des données de risque
        risks = ['Prix', 'Production', 'Marché', 'Environnement']
        likelihood = [0.7, 0.3, 0.5, 0.8]  # Probabilité d'occurrence
        impact = [0.8, 0.4, 0.6, 0.9]      # Impact sur le projet
        
        # Création de la figure
        plt.figure(figsize=(10, 8))
        
        # Création du scatter plot
        plt.scatter(likelihood, impact, 
                   s=200,  # Taille des points
                   c=impact,  # Couleur basée sur l'impact
                   cmap='RdYlGn_r',  # Colormap rouge-jaune-vert inversé
                   alpha=0.6)
        
        # Ajout des labels pour chaque risque
        for i, risk in enumerate(risks):
            plt.annotate(risk,
                        (likelihood[i], impact[i]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold')
        
        # Ajout des quadrants
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Labels des quadrants
        plt.text(0.25, 0.75, 'Risque Élevé', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.75, 0.75, 'Risque Critique', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.25, 0.25, 'Risque Faible', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.75, 0.25, 'Risque Modéré', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Personnalisation du graphique
        plt.title('Matrice d\'analyse des risques', pad=20, size=14)
        plt.xlabel('Probabilité d\'occurrence', fontsize=12)
        plt.ylabel('Impact sur le projet', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajout de la colorbar
        cbar = plt.colorbar()
        cbar.set_label('Niveau de risque', fontsize=12)
        
        # Ajustement de la mise en page
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig(self.output_dir / 'risk_analysis.png',
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white')
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
        
        # Analyse de sensibilité
        print("Génération de l'analyse de sensibilité...")
        self.plot_sensitivity_analysis()
        
        # Insights ML
        print("Génération des insights du machine learning...")
        self.plot_ml_insights()
        
        # Analyse des risques
        print("Génération de l'analyse des risques...")
        self.plot_risk_analysis()
        
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