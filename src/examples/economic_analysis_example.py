import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from models.economic_model import OilFieldParameters, OilFieldEconomicModel

def load_oil_prices(file_path: str) -> pd.Series:
    """Charge les prix du pétrole depuis un fichier CSV."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['price']

def plot_production_profile(production_profile: pd.DataFrame):
    """Visualise le profil de production."""
    plt.figure(figsize=(12, 6))
    plt.plot(production_profile['year'], production_profile['daily_production'], 
             label='Production journalière')
    plt.plot(production_profile['year'], production_profile['cumulative_production'] / 1e6,
             label='Production cumulative (millions de bbl)')
    plt.xlabel('Année')
    plt.ylabel('Production (bbl/jour ou millions de bbl)')
    plt.title('Profil de Production du Champ Pétrolier')
    plt.legend()
    plt.grid(True)
    plt.savefig('production_profile.png')
    plt.close()

def plot_cash_flows(cash_flows: pd.DataFrame):
    """Visualise les flux de trésorerie."""
    plt.figure(figsize=(12, 6))
    plt.plot(cash_flows['year'], cash_flows['revenue'] / 1e6, label='Revenus')
    plt.plot(cash_flows['year'], cash_flows['variable_costs'] / 1e6, label='Coûts variables')
    plt.plot(cash_flows['year'], cash_flows['fixed_costs'] / 1e6, label='Coûts fixes')
    plt.plot(cash_flows['year'], cash_flows['net_cash_flow'] / 1e6, label='Cash Flow Net')
    plt.xlabel('Année')
    plt.ylabel('Montant (millions USD)')
    plt.title('Flux de Trésorerie du Champ Pétrolier')
    plt.legend()
    plt.grid(True)
    plt.savefig('cash_flows.png')
    plt.close()

def plot_sensitivity_analysis(sensitivity_results: dict):
    """Visualise les résultats de l'analyse de sensibilité."""
    n_params = len(sensitivity_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics = ['npv', 'irr', 'payback_period', 'roi']
    metric_names = ['NPV (USD)', 'IRR (%)', 'Payback Period (années)', 'ROI (%)']
    
    for i, (param, results) in enumerate(sensitivity_results.items()):
        for j, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[j]
            ax.plot(results['variation'], results[metric], 
                   label=f'{param}', marker='o')
            ax.set_xlabel('Variation (%)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Sensibilité de {metric_name}')
            ax.grid(True)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')
    plt.close()

def main():
    # Paramètres du champ pétrolier (données réalistes et proches de la norme)
    parameters = OilFieldParameters(
        initial_investment=800e6,  # 800 millions USD
        extraction_cost_per_bbl=25,  # 25 USD/bbl
        fixed_costs_per_year=80e6,  # 80 millions USD/an
        maintenance_cost_per_year=40e6,  # 40 millions USD/an
        environmental_cost_per_year=20e6,  # 20 millions USD/an
        api_gravity=35,  # Pétrole léger
        decline_rate=0.15,  # 15% de déclin annuel
        initial_production=40000,  # 40,000 bbl/jour
        field_lifetime=20,  # 20 ans
        discount_rate=0.1  # 10%
    )

    # Création du modèle
    model = OilFieldEconomicModel(parameters)

    # Génère un tableau de prix (1 prix par année) (données réalistes et proches de la norme)
    oil_prices = (np.random.normal(60, 15, parameters.field_lifetime))  # 20 valeurs

    # Exécution de l'analyse économique
    results = model.run_economic_analysis(oil_prices)

    # Affichage des résultats
    print("\nRésultats de l'analyse économique:")
    print(f"NPV: {results['npv']/1e6:.2f} millions USD")
    print(f"IRR: {results['irr']*100:.2f}%")
    print(f"Payback Period: {results['payback_period']:.2f} années")
    print(f"ROI: {results['roi']:.2f}%")

    # Visualisation des résultats
    plot_production_profile(results['production_profile'])
    plot_cash_flows(results['cash_flows'])

    # Analyse de sensibilité
    sensitivity_params = ['extraction_cost_per_bbl', 'initial_production', 'api_gravity', 'decline_rate']
    variations = [-0.2, -0.1, 0, 0.1, 0.2]  # Variations de ±20%
    sensitivity_results = model.sensitivity_analysis(oil_prices, sensitivity_params, variations)
    plot_sensitivity_analysis(sensitivity_results)

if __name__ == "__main__":
    main() 