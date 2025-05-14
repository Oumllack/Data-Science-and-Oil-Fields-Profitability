import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging

# Ajout du répertoire parent au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_collector import OilDataCollector
from src.models.predictive_models import OilPricePredictor, ProductionDeclinePredictor
from src.models.economic_model import OilFieldEconomicModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Analyse Prédictive des Champs Pétroliers"

# Layout de l'application
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Tableau de Bord d'Analyse Prédictive des Champs Pétroliers",
                   className="text-center my-4"),
            html.P("Analysez la rentabilité des champs pétroliers en utilisant des modèles prédictifs avancés",
                  className="text-center mb-4")
        ])
    ]),
    
    # Paramètres du champ pétrolier
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Paramètres du Champ Pétrolier"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Investissement Initial (millions USD)"),
                            dcc.Input(id="initial-investment", type="number", value=800,
                                    className="form-control mb-3"),
                            html.Label("Coût d'Extraction (USD/bbl)"),
                            dcc.Input(id="extraction-cost", type="number", value=25,
                                    className="form-control mb-3"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Production Initiale (bbl/jour)"),
                            dcc.Input(id="initial-production", type="number", value=50000,
                                    className="form-control mb-3"),
                            html.Label("Taux de Déclin Annuel (%)"),
                            dcc.Input(id="decline-rate", type="number", value=15,
                                    className="form-control mb-3"),
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Coûts Fixes (millions USD/an)"),
                            dcc.Input(id="fixed-costs", type="number", value=80,
                                    className="form-control mb-3"),
                            html.Label("Coûts de Maintenance (millions USD/an)"),
                            dcc.Input(id="maintenance-costs", type="number", value=40,
                                    className="form-control mb-3"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Coûts Environnementaux (millions USD/an)"),
                            dcc.Input(id="environmental-costs", type="number", value=20,
                                    className="form-control mb-3"),
                            html.Label("Taux d'Actualisation (%)"),
                            dcc.Input(id="discount-rate", type="number", value=10,
                                    className="form-control mb-3"),
                        ], width=6)
                    ]),
                    dbc.Button("Lancer l'Analyse", id="run-analysis", 
                             color="primary", className="mt-3 w-100")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Graphiques et résultats
    dbc.Row([
        # Prédiction des prix
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prédiction des Prix du Pétrole"),
                dbc.CardBody([
                    dcc.Graph(id="price-prediction-graph"),
                    html.Div(id="price-metrics", className="mt-3")
                ])
            ], className="mb-4")
        ], width=6),
        
        # Prédiction de la production
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prédiction de la Production"),
                dbc.CardBody([
                    dcc.Graph(id="production-prediction-graph"),
                    html.Div(id="production-metrics", className="mt-3")
                ])
            ], className="mb-4")
        ], width=6)
    ]),
    
    # Métriques économiques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Métriques de Rentabilité"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4("NPV", className="text-center"),
                            html.H3(id="npv-value", className="text-center text-primary")
                        ], width=3),
                        dbc.Col([
                            html.H4("IRR", className="text-center"),
                            html.H3(id="irr-value", className="text-center text-primary")
                        ], width=3),
                        dbc.Col([
                            html.H4("Payback", className="text-center"),
                            html.H3(id="payback-value", className="text-center text-primary")
                        ], width=3),
                        dbc.Col([
                            html.H4("ROI", className="text-center"),
                            html.H3(id="roi-value", className="text-center text-primary")
                        ], width=3)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Analyse de sensibilité
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analyse de Sensibilité"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Paramètre à Analyser"),
                            dcc.Dropdown(
                                id="sensitivity-parameter",
                                options=[
                                    {'label': 'Prix du Pétrole', 'value': 'price'},
                                    {'label': 'Production Initiale', 'value': 'production'},
                                    {'label': 'Coût d\'Extraction', 'value': 'extraction_cost'},
                                    {'label': 'Taux d\'Actualisation', 'value': 'discount_rate'}
                                ],
                                value='price',
                                className="mb-3"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Variation (%)"),
                            dcc.Slider(
                                id="sensitivity-range",
                                min=-30,
                                max=30,
                                step=5,
                                value=0,
                                marks={i: f'{i}%' for i in range(-30, 31, 5)},
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    dcc.Graph(id="sensitivity-graph")
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output("price-prediction-graph", "figure"),
     Output("production-prediction-graph", "figure"),
     Output("price-metrics", "children"),
     Output("production-metrics", "children"),
     Output("npv-value", "children"),
     Output("irr-value", "children"),
     Output("payback-value", "children"),
     Output("roi-value", "children")],
    [Input("run-analysis", "n_clicks")],
    [State("initial-investment", "value"),
     State("extraction-cost", "value"),
     State("initial-production", "value"),
     State("decline-rate", "value"),
     State("fixed-costs", "value"),
     State("maintenance-costs", "value"),
     State("environmental-costs", "value"),
     State("discount-rate", "value")]
)
def update_analysis(n_clicks, initial_investment, extraction_cost, initial_production,
                   decline_rate, fixed_costs, maintenance_costs, environmental_costs,
                   discount_rate):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Initialisation des modèles
    collector = OilDataCollector()
    
    # Récupération des données historiques
    historical_data = collector.prepare_training_data(
        start_date='2010-01-01',
        end_date='2023-12-31'
    )
    
    # Prédiction des prix
    price_predictor = OilPricePredictor(model_type='xgboost')
    price_metrics = price_predictor.train(historical_data)
    
    future_dates = pd.date_range(
        start='2024-01-01',
        end='2025-12-31',
        freq='D'
    )
    future_data = historical_data.iloc[-len(future_dates):].copy()
    future_data.index = future_dates
    
    price_predictions = price_predictor.predict(future_data)
    
    # Prédiction de la production
    training_production = collector.get_field_production_data(
        field_name='Training Field',
        decline_rate=decline_rate/100,
        initial_production=initial_production,
        years=5
    )
    
    decline_predictor = ProductionDeclinePredictor()
    production_metrics = decline_predictor.train(training_production)
    
    future_production = decline_predictor.predict_decline(
        initial_production=initial_production,
        days=365*20
    )
    
    # Analyse économique
    economic_model = OilFieldEconomicModel(
        initial_investment=initial_investment * 1e6,
        extraction_cost=extraction_cost,
        fixed_costs=fixed_costs * 1e6,
        maintenance_costs=maintenance_costs * 1e6,
        environmental_costs=environmental_costs * 1e6,
        discount_rate=discount_rate/100
    )
    
    cash_flows = economic_model.calculate_cash_flows(
        production_profile=future_production['daily_production'].values,
        oil_prices=price_predictions.values
    )
    
    npv = economic_model.calculate_npv(cash_flows)
    irr = economic_model.calculate_irr(cash_flows)
    payback = economic_model.calculate_payback_period(cash_flows)
    roi = economic_model.calculate_roi(cash_flows)
    
    # Création des graphiques
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['price'],
        name='Prix historiques',
        line=dict(color='blue')
    ))
    price_fig.add_trace(go.Scatter(
        x=price_predictions.index,
        y=price_predictions,
        name='Prédictions',
        line=dict(color='red', dash='dash')
    ))
    price_fig.update_layout(
        title="Prédiction des Prix du Pétrole",
        xaxis_title="Date",
        yaxis_title="Prix (USD/bbl)",
        showlegend=True
    )
    
    production_fig = go.Figure()
    production_fig.add_trace(go.Scatter(
        x=training_production['date'],
        y=training_production['daily_production'],
        name='Production historique',
        line=dict(color='blue')
    ))
    production_fig.add_trace(go.Scatter(
        x=future_production['date'],
        y=future_production['daily_production'],
        name='Production prédite',
        line=dict(color='red', dash='dash')
    ))
    production_fig.update_layout(
        title="Prédiction de la Production",
        xaxis_title="Date",
        yaxis_title="Production (bbl/jour)",
        showlegend=True
    )
    
    # Métriques formatées
    price_metrics_html = html.Div([
        html.H5("Métriques du Modèle de Prix"),
        html.P(f"RMSE: {price_metrics['rmse']:.2f} USD/bbl"),
        html.P(f"R²: {price_metrics['r2']:.2f}")
    ])
    
    production_metrics_html = html.Div([
        html.H5("Métriques du Modèle de Production"),
        html.P(f"RMSE: {production_metrics['rmse']:.2f} bbl/jour"),
        html.P(f"R²: {production_metrics['r2']:.2f}")
    ])
    
    return (price_fig, production_fig, price_metrics_html, production_metrics_html,
            f"{npv/1e6:.1f}M USD", f"{irr*100:.1f}%",
            f"{payback:.1f} ans", f"{roi*100:.1f}%")

@app.callback(
    Output("sensitivity-graph", "figure"),
    [Input("sensitivity-parameter", "value"),
     Input("sensitivity-range", "value"),
     Input("run-analysis", "n_clicks")],
    [State("initial-investment", "value"),
     State("extraction-cost", "value"),
     State("initial-production", "value"),
     State("decline-rate", "value"),
     State("fixed-costs", "value"),
     State("maintenance-costs", "value"),
     State("environmental-costs", "value"),
     State("discount-rate", "value")]
)
def update_sensitivity_analysis(parameter, variation, n_clicks, *args):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Paramètres de base
    params = {
        'initial_investment': args[0] * 1e6,
        'extraction_cost': args[1],
        'initial_production': args[2],
        'decline_rate': args[3]/100,
        'fixed_costs': args[4] * 1e6,
        'maintenance_costs': args[5] * 1e6,
        'environmental_costs': args[6] * 1e6,
        'discount_rate': args[7]/100
    }
    
    # Calcul des variations
    variations = np.linspace(-variation/100, variation/100, 11)
    npv_values = []
    
    for var in variations:
        # Copie des paramètres
        current_params = params.copy()
        
        # Application de la variation
        if parameter == 'price':
            # Simulation d'une variation des prix
            price_multiplier = 1 + var
            current_params['extraction_cost'] *= price_multiplier
        elif parameter == 'production':
            current_params['initial_production'] *= (1 + var)
        elif parameter == 'extraction_cost':
            current_params['extraction_cost'] *= (1 + var)
        elif parameter == 'discount_rate':
            current_params['discount_rate'] *= (1 + var)
        
        # Calcul du NPV avec les paramètres modifiés
        collector = OilDataCollector()
        historical_data = collector.prepare_training_data(
            start_date='2010-01-01',
            end_date='2023-12-31'
        )
        
        price_predictor = OilPricePredictor(model_type='xgboost')
        price_predictor.train(historical_data)
        
        future_dates = pd.date_range(
            start='2024-01-01',
            end='2025-12-31',
            freq='D'
        )
        future_data = historical_data.iloc[-len(future_dates):].copy()
        future_data.index = future_dates
        
        price_predictions = price_predictor.predict(future_data)
        if parameter == 'price':
            price_predictions *= price_multiplier
        
        training_production = collector.get_field_production_data(
            field_name='Training Field',
            decline_rate=current_params['decline_rate'],
            initial_production=current_params['initial_production'],
            years=5
        )
        
        decline_predictor = ProductionDeclinePredictor()
        decline_predictor.train(training_production)
        
        future_production = decline_predictor.predict_decline(
            initial_production=current_params['initial_production'],
            days=365*20
        )
        
        economic_model = OilFieldEconomicModel(
            initial_investment=current_params['initial_investment'],
            extraction_cost=current_params['extraction_cost'],
            fixed_costs=current_params['fixed_costs'],
            maintenance_costs=current_params['maintenance_costs'],
            environmental_costs=current_params['environmental_costs'],
            discount_rate=current_params['discount_rate']
        )
        
        cash_flows = economic_model.calculate_cash_flows(
            production_profile=future_production['daily_production'].values,
            oil_prices=price_predictions.values
        )
        
        npv = economic_model.calculate_npv(cash_flows)
        npv_values.append(npv/1e6)  # Conversion en millions USD
    
    # Création du graphique de sensibilité
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=variations*100,  # Conversion en pourcentage
        y=npv_values,
        mode='lines+markers',
        name='NPV'
    ))
    
    fig.update_layout(
        title=f"Analyse de Sensibilité - Impact sur le NPV",
        xaxis_title=f"Variation de {parameter} (%)",
        yaxis_title="NPV (millions USD)",
        showlegend=True
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 