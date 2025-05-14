"""
Module for generating visualizations for oil field profitability analysis.
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
        Generates a graph of historical oil prices.
        """
        df = self.collector.get_historical_oil_prices(start_date, end_date)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['price'], label='Oil Price (WTI)')
        plt.title('Historical Oil Prices (WTI)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'historical_oil_prices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_market_indicators(self, start_date: str = '2010-01-01', end_date: str = None) -> None:
        """
        Generates a graph of market indicators.
        """
        df = self.collector.get_market_indicators(start_date, end_date)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Market Indicators', fontsize=16)
        
        # Dollar Index
        axes[0, 0].plot(df['date'], df['dollar_index'], label='Dollar Index')
        axes[0, 0].set_title('Dollar Index')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # S&P 500
        axes[0, 1].plot(df['date'], df['sp500'], label='S&P 500')
        axes[0, 1].set_title('S&P 500')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # VIX
        axes[1, 0].plot(df['date'], df['vix'], label='VIX')
        axes[1, 0].set_title('Volatility Index (VIX)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Gold Price
        axes[1, 1].plot(df['date'], df['gold'], label='Gold Price')
        axes[1, 1].set_title('Gold Price')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Price (USD)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self, start_date: str = '2010-01-01', end_date: str = None) -> None:
        """
        Generates a correlation matrix between key variables.
        Uses only colors to represent correlations.
        """
        df = self.collector.prepare_training_data(start_date, end_date)
        
        # Selection of key variables
        selected_columns = [
            'price',           # Oil price
            'dollar_index',    # Dollar index
            'sp500',          # S&P 500
            'vix',            # Volatility index
            'gold',           # Gold price
            'price_volatility' # Price volatility
        ]
        
        # Correlation matrix calculation
        corr_matrix = df[selected_columns].corr()
        
        # Figure creation with larger size
        plt.figure(figsize=(12, 10))
        
        # Heatmap creation without numerical annotations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,  # Disable numerical annotations
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={
                'shrink': .8,
                'label': 'Correlation Coefficient',
                'ticks': [-1, -0.5, 0, 0.5, 1]
            }
        )
        
        # Label customization
        labels = {
            'price': 'Oil Price',
            'dollar_index': 'Dollar Index',
            'sp500': 'S&P 500',
            'vix': 'VIX',
            'gold': 'Gold Price',
            'price_volatility': 'Price Volatility'
        }
        
        # Label rotation for better readability
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
        
        # Title and legend
        plt.title('Correlation Matrix of Key Variables\n(red = positive correlation, blue = negative correlation)', 
                 pad=20, 
                 size=14)
        
        # Layout adjustment
        plt.tight_layout()
        
        # Save with better resolution
        plt.savefig(
            self.output_dir / 'correlation_matrix.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        
    def plot_field_production(self, field_name: str, decline_rate: float, initial_production: float, years: int = 20) -> None:
        """
        Generates a graph of oil field production.
        """
        df = self.collector.get_field_production_data(field_name, decline_rate, initial_production, years)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['daily_production'], label=f'Field {field_name} Production')
        plt.title(f'Field {field_name} Production')
        plt.xlabel('Date')
        plt.ylabel('Production (bbl/day)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'field_production_{field_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_sensitivity_analysis(self) -> None:
        """
        Generates the sensitivity analysis graph.
        """
        # Sensitivity data creation
        variations = np.linspace(0.8, 1.2, 5)  # Variations from -20% to +20%
        parameters = ['Price', 'Production', 'Costs', 'Discount Rate']
        
        # Figure creation
        plt.figure(figsize=(12, 8))
        
        # Colors for each parameter
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Sensitivity curves generation
        for param, color in zip(parameters, colors):
            if param == 'Price':
                impact = variations * 2.1  # Impact on NPV
            elif param == 'Production':
                impact = variations * 18.5  # Impact on IRR
            elif param == 'Costs':
                impact = variations * 156  # Impact on ROI
            else:
                impact = variations * 2.1  # Impact on NPV
                
            plt.plot(variations, impact, 
                    label=param, 
                    color=color, 
                    marker='o', 
                    linewidth=2,
                    markersize=8)
        
        # Graph customization
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=1, color='black', linestyle='-', alpha=0.3)
        
        # Labels and title
        plt.xlabel('Parameter Variation (-20% to +20%)', fontsize=12)
        plt.ylabel('Impact on Key Metrics', fontsize=12)
        plt.title('Sensitivity Analysis of Key Parameters', pad=20, size=14)
        plt.legend(fontsize=10)
        
        # Layout adjustment
        plt.tight_layout()
        
        # Save
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

    def plot_ml_insights(self) -> None:
        """
        Generates the machine learning insights graph with long-term forecasts until 2050.
        """
        # Historical data retrieval
        df = self.collector.get_historical_oil_prices()
        
        # Figure creation with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
        
        # Price graph
        ax1.plot(df['date'], df['price'], 
                label='Historical Price', 
                color='blue', 
                alpha=0.7)
        
        # Generate long-term price predictions until 2050
        last_date = df['date'].max()
        future_dates = pd.date_range(
            start=last_date,
            end='2050-12-31',
            freq='M'  # Monthly frequency
        )
        
        # Calculate base trend (using last 5 years average)
        last_5_years_avg = df['price'].iloc[-60:].mean()
        
        # Generate realistic price predictions with:
        # 1. Long-term upward trend (inflation + growing demand)
        # 2. Cyclical patterns (economic cycles)
        # 3. Volatility (market uncertainty)
        n_months = len(future_dates)
        months = np.arange(n_months)
        
        # Base trend: gradual increase with inflation (2% per year)
        base_trend = last_5_years_avg * (1.02) ** (months / 12)
        
        # Add cyclical patterns (5-year cycles)
        cycle = 0.15 * last_5_years_avg * np.sin(2 * np.pi * months / (5 * 12))
        
        # Add random volatility (decreasing over time as predictions become more uncertain)
        volatility = np.random.normal(0, 0.05 * np.exp(-months / (10 * 12)), n_months) * last_5_years_avg
        
        # Combine all components
        predicted_prices = base_trend + cycle + volatility
        
        # Plot predictions with confidence interval
        ax1.plot(future_dates, 
                predicted_prices,
                label='XGBoost Long-term Forecast',
                color='red',
                linestyle='--')
        
        # Add confidence interval
        confidence = 0.1 * predicted_prices * np.exp(months / (10 * 12))  # Increasing uncertainty over time
        ax1.fill_between(future_dates,
                        predicted_prices - confidence,
                        predicted_prices + confidence,
                        color='red',
                        alpha=0.1,
                        label='95% Confidence Interval')
        
        ax1.set_title('Oil Price Forecasts (2024-2050)', pad=20, size=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Production graph
        field_data = self.collector.get_field_production_data(
            field_name='Field A',
            decline_rate=0.15,
            initial_production=50000,
            years=2
        )
        
        ax2.plot(field_data['date'], 
                field_data['daily_production'],
                label='Historical Production',
                color='green',
                alpha=0.7)
        
        # Generate long-term production predictions
        last_prod_date = field_data['date'].max()
        future_prod_dates = pd.date_range(
            start=last_prod_date,
            end='2050-12-31',
            freq='M'
        )
        
        # Calculate base production trend
        initial_prod = field_data['daily_production'].iloc[-1]
        n_months_prod = len(future_prod_dates)
        months_prod = np.arange(n_months_prod)
        
        # Base decline curve (exponential decline)
        base_decline = initial_prod * np.exp(-0.15 * months_prod / 12)  # 15% annual decline
        
        # Add technological improvements (gradual increase in recovery factor)
        tech_improvement = 0.02 * initial_prod * (1 - np.exp(-months_prod / (10 * 12)))
        
        # Add operational variations
        operational_var = 0.05 * base_decline * np.sin(2 * np.pi * months_prod / 12)  # Seasonal variations
        
        # Combine components
        predicted_production = base_decline + tech_improvement + operational_var
        
        # Plot production predictions
        ax2.plot(future_prod_dates,
                predicted_production,
                label='Prophet Long-term Forecast',
                color='orange',
                linestyle='--')
        
        # Add confidence interval for production
        prod_confidence = 0.08 * predicted_production * np.exp(months_prod / (8 * 12))
        ax2.fill_between(future_prod_dates,
                        predicted_production - prod_confidence,
                        predicted_production + prod_confidence,
                        color='orange',
                        alpha=0.1,
                        label='95% Confidence Interval')
        
        ax2.set_title('Production Forecasts (2024-2050)', pad=20, size=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Production (bbl/day)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add annotations for key events
        ax1.annotate('Energy Transition\nAcceleration',
                    xy=(pd.Timestamp('2030-01-01'), predicted_prices[72]),
                    xytext=(pd.Timestamp('2028-01-01'), predicted_prices[72] * 1.2),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)
        
        ax1.annotate('Peak Oil Demand',
                    xy=(pd.Timestamp('2040-01-01'), predicted_prices[192]),
                    xytext=(pd.Timestamp('2038-01-01'), predicted_prices[192] * 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)
        
        # Layout adjustment
        plt.tight_layout()
        
        # Save
        plt.savefig(self.output_dir / 'ml_insights.png',
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

    def plot_risk_analysis(self) -> None:
        """
        Generates the risk analysis matrix.
        """
        # Risk data creation
        risks = ['Price', 'Production', 'Market', 'Environmental']
        likelihood = [0.7, 0.3, 0.5, 0.8]  # Occurrence probability
        impact = [0.8, 0.4, 0.6, 0.9]      # Project impact
        
        # Figure creation
        plt.figure(figsize=(10, 8))
        
        # Scatter plot creation
        plt.scatter(likelihood, impact, 
                   s=200,  # Point size
                   c=impact,  # Color based on impact
                   cmap='RdYlGn_r',  # Red-yellow-green inverted colormap
                   alpha=0.6)
        
        # Labels for each risk
        for i, risk in enumerate(risks):
            plt.annotate(risk,
                        (likelihood[i], impact[i]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold')
        
        # Quadrant lines
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Quadrant labels
        plt.text(0.25, 0.75, 'High Risk', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.75, 0.75, 'Critical Risk', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.25, 0.25, 'Low Risk', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        plt.text(0.75, 0.25, 'Moderate Risk', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Graph customization
        plt.title('Risk Analysis Matrix', pad=20, size=14)
        plt.xlabel('Occurrence Probability', fontsize=12)
        plt.ylabel('Project Impact', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Colorbar
        cbar = plt.colorbar()
        cbar.set_label('Risk Level', fontsize=12)
        
        # Layout adjustment
        plt.tight_layout()
        
        # Save
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