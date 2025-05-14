import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class OilFieldParameters:
    """Paramètres d'un champ pétrolier pour l'analyse économique."""
    initial_investment: float  # Investissement initial en USD
    extraction_cost_per_bbl: float  # Coût d'extraction par baril en USD
    fixed_costs_per_year: float  # Coûts fixes annuels en USD
    maintenance_cost_per_year: float  # Coûts de maintenance annuels en USD
    environmental_cost_per_year: float  # Coûts environnementaux annuels en USD
    api_gravity: float  # Densité API du pétrole
    decline_rate: float  # Taux de déclin de production annuel (%)
    initial_production: float  # Production initiale (bbl/jour)
    field_lifetime: int  # Durée de vie du champ en années
    discount_rate: float  # Taux d'actualisation (%)

class OilFieldEconomicModel:
    def __init__(self, parameters: OilFieldParameters):
        self.params = parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validation des paramètres d'entrée."""
        assert 0 < self.params.api_gravity <= 45, "La densité API doit être entre 0 et 45"
        assert 0 <= self.params.decline_rate <= 1, "Le taux de déclin doit être entre 0 et 1"
        assert self.params.initial_production > 0, "La production initiale doit être positive"
        assert self.params.field_lifetime > 0, "La durée de vie du champ doit être positive"
        assert 0 <= self.params.discount_rate <= 1, "Le taux d'actualisation doit être entre 0 et 1"

    def calculate_production_profile(self) -> pd.DataFrame:
        """Calcule le profil de production sur la durée de vie du champ."""
        years = range(self.params.field_lifetime)
        daily_production = []
        cumulative_production = 0

        for year in years:
            # Calcul de la production journalière avec déclin exponentiel
            daily_prod = self.params.initial_production * (1 - self.params.decline_rate) ** year
            daily_production.append(daily_prod)
            cumulative_production += daily_prod * 365

        return pd.DataFrame({
            'year': years,
            'daily_production': daily_production,
            'annual_production': [p * 365 for p in daily_production],
            'cumulative_production': [sum(daily_production[:i+1]) * 365 for i in range(len(years))]
        })

    def calculate_cash_flows(self, oil_prices: pd.Series) -> pd.DataFrame:
        """Calcule les flux de trésorerie sur la durée de vie du champ."""
        production = self.calculate_production_profile()
        
        # S'assurer que oil_prices est de la bonne longueur et sous forme de tableau
        oil_prices = np.array(oil_prices)
        if len(oil_prices) != self.params.field_lifetime:
            raise ValueError(f"La longueur de oil_prices ({len(oil_prices)}) doit être égale à la durée de vie du champ ({self.params.field_lifetime})")
        
        # Ajustement du prix en fonction de la qualité (API gravity)
        quality_factor = 1 + (self.params.api_gravity - 30) * 0.01  # Ajustement basé sur API 30
        adjusted_prices = oil_prices * quality_factor

        # Calcul des revenus et coûts
        revenues = production['annual_production'] * adjusted_prices
        variable_costs = production['annual_production'] * self.params.extraction_cost_per_bbl
        fixed_costs = np.full(self.params.field_lifetime, self.params.fixed_costs_per_year)
        maintenance_costs = np.full(self.params.field_lifetime, self.params.maintenance_cost_per_year)
        environmental_costs = np.full(self.params.field_lifetime, self.params.environmental_cost_per_year)

        # Calcul des flux de trésorerie
        cash_flows = pd.DataFrame({
            'year': production['year'],
            'revenue': revenues,
            'variable_costs': variable_costs,
            'fixed_costs': fixed_costs,
            'maintenance_costs': maintenance_costs,
            'environmental_costs': environmental_costs,
            'net_cash_flow': revenues - variable_costs - fixed_costs - maintenance_costs - environmental_costs
        })

        # Ajout de l'investissement initial
        cash_flows.loc[0, 'net_cash_flow'] -= self.params.initial_investment

        return cash_flows

    def calculate_npv(self, cash_flows: pd.DataFrame) -> float:
        """Calcule la Valeur Actuelle Nette (NPV)."""
        discount_factors = 1 / (1 + self.params.discount_rate) ** cash_flows['year']
        return (cash_flows['net_cash_flow'] * discount_factors).sum()

    def calculate_irr(self, cash_flows: pd.DataFrame) -> float:
        """Calcule le Taux de Rentabilité Interne (IRR)."""
        from scipy.optimize import newton
        
        def npv_function(rate):
            discount_factors = 1 / (1 + rate) ** cash_flows['year']
            return (cash_flows['net_cash_flow'] * discount_factors).sum()

        try:
            return newton(npv_function, x0=0.1)  # Commence avec un taux de 10%
        except:
            return np.nan

    def calculate_payback_period(self, cash_flows: pd.DataFrame) -> float:
        """Calcule la période de récupération (Payback Period)."""
        cumulative_cash_flow = cash_flows['net_cash_flow'].cumsum()
        if cumulative_cash_flow.iloc[-1] < 0:
            return np.nan
        
        # Trouve l'année où le cash flow cumulé devient positif
        payback_year = cumulative_cash_flow[cumulative_cash_flow >= 0].index[0]
        if payback_year == 0:
            return 0
        
        # Calcul précis de la période de récupération
        last_negative = cumulative_cash_flow.iloc[payback_year - 1]
        first_positive = cumulative_cash_flow.iloc[payback_year]
        fraction = abs(last_negative) / (first_positive - last_negative)
        
        return payback_year - 1 + fraction

    def calculate_roi(self, cash_flows: pd.DataFrame) -> float:
        """Calcule le Return on Investment (ROI)."""
        total_profit = cash_flows['net_cash_flow'].sum()
        return (total_profit / self.params.initial_investment) * 100

    def run_economic_analysis(self, oil_prices: pd.Series) -> Dict:
        """Exécute l'analyse économique complète du champ pétrolier."""
        cash_flows = self.calculate_cash_flows(oil_prices)
        
        return {
            'npv': self.calculate_npv(cash_flows),
            'irr': self.calculate_irr(cash_flows),
            'payback_period': self.calculate_payback_period(cash_flows),
            'roi': self.calculate_roi(cash_flows),
            'cash_flows': cash_flows,
            'production_profile': self.calculate_production_profile()
        }

    def sensitivity_analysis(self, 
                           oil_prices: pd.Series,
                           parameters: List[str],
                           variations: List[float]) -> Dict[str, pd.DataFrame]:
        """Effectue une analyse de sensibilité sur les paramètres spécifiés."""
        base_results = self.run_economic_analysis(oil_prices)
        sensitivity_results = {}

        for param in parameters:
            results = []
            base_value = getattr(self.params, param)
            
            for variation in variations:
                # Crée une copie des paramètres avec la variation
                modified_params = OilFieldParameters(**{
                    **self.params.__dict__,
                    param: base_value * (1 + variation)
                })
                
                # Calcule les résultats avec les paramètres modifiés
                modified_model = OilFieldEconomicModel(modified_params)
                modified_results = modified_model.run_economic_analysis(oil_prices)
                
                results.append({
                    'variation': variation * 100,  # En pourcentage
                    'npv': modified_results['npv'],
                    'irr': modified_results['irr'],
                    'payback_period': modified_results['payback_period'],
                    'roi': modified_results['roi']
                })
            
            sensitivity_results[param] = pd.DataFrame(results)

        return sensitivity_results 