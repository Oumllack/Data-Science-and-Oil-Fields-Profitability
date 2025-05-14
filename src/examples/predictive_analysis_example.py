import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Ajout du répertoire parent au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_collector import OilDataCollector
from src.models.predictive_models import OilPricePredictor, ProductionDeclinePredictor
from src.models.economic_model import OilFieldEconomicModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_price_predictions(historical_data: pd.DataFrame, 
                         predictions: pd.Series,
                         title: str = "Prédiction des prix du pétrole"):
    """Visualise les prix historiques et les prédictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data['price'], 
             label='Prix historiques', color='blue')
    plt.plot(predictions.index, predictions, 
             label='Prédictions', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Prix (USD/bbl)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_predictions.png')
    plt.close()

def plot_production_decline(historical_data: pd.DataFrame,
                          predictions: pd.DataFrame,
                          title: str = "Prédiction du déclin de production"):
    """Visualise le déclin de production historique et prédit."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['date'], historical_data['daily_production'],
             label='Production historique', color='blue')
    plt.plot(predictions['date'], predictions['daily_production'],
             label='Production prédite', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Production (bbl/jour)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('production_decline.png')
    plt.close()

def plot_feature_importance(importance_df: pd.DataFrame,
                          title: str = "Importance des features"):
    """Visualise l'importance des features pour le modèle de prix."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10),
                x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    """Exemple d'utilisation des modèles prédictifs."""
    # Initialisation du collecteur de données
    collector = OilDataCollector()
    
    # Récupération des données historiques
    logger.info("Récupération des données historiques...")
    historical_data = collector.prepare_training_data(
        start_date='2010-01-01',
        end_date='2023-12-31'
    )
    
    # 1. Prédiction des prix du pétrole
    logger.info("Entraînement du modèle de prédiction des prix...")
    price_predictor = OilPricePredictor(model_type='xgboost')
    metrics = price_predictor.train(historical_data)
    
    logger.info(f"Métriques du modèle de prix:")
    for metric, value in metrics.items():
        if metric != 'feature_importance':
            logger.info(f"{metric}: {value:.4f}")
    
    # Visualisation de l'importance des features
    if 'feature_importance' in metrics:
        plot_feature_importance(metrics['feature_importance'])
    
    # Prédiction des prix futurs
    future_dates = pd.date_range(
        start='2024-01-01',
        end='2025-12-31',
        freq='D'
    )
    future_data = historical_data.iloc[-len(future_dates):].copy()
    future_data.index = future_dates
    
    price_predictions = price_predictor.predict(future_data)
    plot_price_predictions(historical_data, price_predictions)
    
    # 2. Prédiction du déclin de production
    logger.info("\nEntraînement du modèle de déclin de production...")
    
    # Génération de données de production synthétiques pour l'entraînement
    training_production = collector.get_field_production_data(
        field_name='Training Field',
        decline_rate=0.15,  # 15% de déclin annuel
        initial_production=40000,  # 40,000 bbl/jour
        years=5  # 5 ans de données d'entraînement
    )
    
    decline_predictor = ProductionDeclinePredictor()
    decline_metrics = decline_predictor.train(training_production)
    
    logger.info(f"Métriques du modèle de déclin:")
    for metric, value in decline_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Prédiction du déclin pour un nouveau champ
    future_production = decline_predictor.predict_decline(
        initial_production=50000,  # 50,000 bbl/jour
        days=365*20  # 20 ans
    )
    plot_production_decline(training_production, future_production)
    
    # 3. Analyse économique combinée
    logger.info("\nAnalyse économique avec les prédictions...")
    
    # Création d'un modèle économique avec les prédictions
    economic_model = OilFieldEconomicModel(
        initial_investment=800_000_000,  # 800 millions USD
        extraction_cost=25,  # 25 USD/bbl
        fixed_costs=80_000_000,  # 80 millions USD/an
        maintenance_costs=40_000_000,  # 40 millions USD/an
        environmental_costs=20_000_000,  # 20 millions USD/an
        discount_rate=0.1  # 10%
    )
    
    # Calcul des flux de trésorerie avec les prédictions
    cash_flows = economic_model.calculate_cash_flows(
        production_profile=future_production['daily_production'].values,
        oil_prices=price_predictions.values
    )
    
    # Calcul des métriques de rentabilité
    npv = economic_model.calculate_npv(cash_flows)
    irr = economic_model.calculate_irr(cash_flows)
    payback = economic_model.calculate_payback_period(cash_flows)
    roi = economic_model.calculate_roi(cash_flows)
    
    logger.info("\nRésultats de l'analyse économique avec prédictions:")
    logger.info(f"NPV: {npv/1e6:.2f} millions USD")
    logger.info(f"IRR: {irr*100:.2f}%")
    logger.info(f"Payback Period: {payback:.2f} années")
    logger.info(f"ROI: {roi*100:.2f}%")
    
    # Sauvegarde des modèles
    price_predictor.save_model('models/price_predictor.joblib')
    decline_predictor.save_model('models/decline_predictor.joblib')
    
    logger.info("\nAnalyse terminée. Les visualisations ont été sauvegardées.")

if __name__ == "__main__":
    main() 