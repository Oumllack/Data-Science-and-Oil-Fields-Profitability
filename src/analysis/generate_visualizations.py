"""
Generate all visualizations for the scientific analysis report.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from src.data.data_collector import OilDataCollector

class VisualizationGenerator:
    def __init__(self):
        self.output_dir = Path('docs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = OilDataCollector()
        
    def generate_all_visualizations(self):
        """Generate all visualizations for the report."""
        self.generate_price_analysis()
        self.generate_production_analysis()
        self.generate_economic_metrics()
        self.generate_sensitivity_analysis()
        self.generate_ml_insights()
        self.generate_risk_analysis()
        
    def generate_price_analysis(self):
        """Generate historical price analysis visualization."""
        # Get historical prices
        prices_df = self.collector.get_historical_oil_prices()
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=prices_df['date'],
            y=prices_df['price'],
            name='WTI Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=prices_df['date'],
            y=prices_df['price'].rolling(window=30).mean(),
            name='30-day MA',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=prices_df['date'],
            y=prices_df['price'].rolling(window=90).mean(),
            name='90-day MA',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Historical WTI Oil Prices (2010-2024)',
            xaxis_title='Date',
            yaxis_title='Price (USD/bbl)',
            template='plotly_white',
            showlegend=True
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'historical_prices.html')
        fig.write_image(self.output_dir / 'historical_prices.png')
        
    def generate_production_analysis(self):
        """Generate production decline analysis visualization."""
        # Generate synthetic production data
        field_data = self.collector.get_field_production_data(
            field_name='Sample Field',
            decline_rate=0.15,
            initial_production=50000,
            years=20
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add production line
        fig.add_trace(go.Scatter(
            x=field_data['date'],
            y=field_data['daily_production'],
            name='Daily Production',
            line=dict(color='blue', width=2)
        ))
        
        # Add theoretical decline curve
        theoretical_decline = 50000 * (1 - 0.15) ** (np.arange(len(field_data)) / 365)
        fig.add_trace(go.Scatter(
            x=field_data['date'],
            y=theoretical_decline,
            name='Theoretical Decline',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Production Decline Analysis',
            xaxis_title='Date',
            yaxis_title='Production (bbl/day)',
            template='plotly_white',
            showlegend=True
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'production_decline.html')
        fig.write_image(self.output_dir / 'production_decline.png')
        
    def generate_economic_metrics(self):
        """Generate economic metrics visualization."""
        # Create sample data for different scenarios
        scenarios = ['Base Case', 'Optimistic', 'Pessimistic']
        npv = [2.1, 3.2, 0.8]  # in billions
        irr = [18.5, 25.3, 12.1]  # in percent
        payback = [4.2, 3.1, 5.8]  # in years
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('NPV (Billions USD)', 'IRR (%)', 'Payback Period (Years)')
        )
        
        # Add NPV bars
        fig.add_trace(
            go.Bar(x=scenarios, y=npv, name='NPV'),
            row=1, col=1
        )
        
        # Add IRR bars
        fig.add_trace(
            go.Bar(x=scenarios, y=irr, name='IRR'),
            row=1, col=2
        )
        
        # Add Payback bars
        fig.add_trace(
            go.Bar(x=scenarios, y=payback, name='Payback'),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Economic Metrics by Scenario',
            template='plotly_white',
            showlegend=False,
            height=500
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'economic_metrics.html')
        fig.write_image(self.output_dir / 'economic_metrics.png')
        
    def generate_sensitivity_analysis(self):
        """Generate sensitivity analysis visualization."""
        # Create sample data for sensitivity analysis
        variations = np.linspace(0.8, 1.2, 5)
        parameters = ['Price', 'Production', 'Cost', 'Discount Rate']
        
        # Create figure
        fig = go.Figure()
        
        # Add lines for each parameter
        for param in parameters:
            # Generate sample sensitivity data
            if param == 'Price':
                impact = variations * 2.1  # NPV impact
            elif param == 'Production':
                impact = variations * 18.5  # IRR impact
            elif param == 'Cost':
                impact = variations * 156  # ROI impact
            else:
                impact = variations * 2.1  # NPV impact
                
            fig.add_trace(go.Scatter(
                x=variations,
                y=impact,
                name=param,
                mode='lines+markers'
            ))
        
        # Update layout
        fig.update_layout(
            title='Sensitivity Analysis',
            xaxis_title='Parameter Variation',
            yaxis_title='Impact on Key Metrics',
            template='plotly_white',
            showlegend=True
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'sensitivity_analysis.html')
        fig.write_image(self.output_dir / 'sensitivity_analysis.png')
        
    def generate_ml_insights(self):
        """Generate machine learning insights visualization."""
        # Get historical data
        prices_df = self.collector.get_historical_oil_prices()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Forecasting', 'Production Forecasting')
        )
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=prices_df['date'],
                y=prices_df['price'],
                name='Historical Prices',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add predicted prices (sample data)
        future_dates = pd.date_range(
            start=prices_df['date'].max(),
            periods=90,
            freq='D'
        )
        predicted_prices = prices_df['price'].iloc[-90:].values * np.random.normal(1, 0.02, 90)
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predicted_prices,
                name='Predicted Prices',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add production forecast (sample data)
        field_data = self.collector.get_field_production_data(
            field_name='Sample Field',
            decline_rate=0.15,
            initial_production=50000,
            years=5
        )
        
        fig.add_trace(
            go.Scatter(
                x=field_data['date'],
                y=field_data['daily_production'],
                name='Historical Production',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Add predicted production
        future_production = field_data['daily_production'].iloc[-365:].values * np.random.normal(1, 0.05, 365)
        future_dates_prod = pd.date_range(
            start=field_data['date'].max(),
            periods=365,
            freq='D'
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates_prod,
                y=future_production,
                name='Predicted Production',
                line=dict(color='orange', dash='dash')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Machine Learning Forecasting Insights',
            template='plotly_white',
            showlegend=True,
            height=800
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'ml_insights.html')
        fig.write_image(self.output_dir / 'ml_insights.png')
        
    def generate_risk_analysis(self):
        """Generate risk analysis visualization."""
        # Create sample risk data
        risks = ['Price', 'Production', 'Market', 'Environmental']
        likelihood = [0.7, 0.3, 0.5, 0.8]
        impact = [0.8, 0.4, 0.6, 0.9]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=likelihood,
            y=impact,
            mode='markers+text',
            marker=dict(
                size=20,
                color=impact,
                colorscale='RdYlGn_r',
                showscale=True
            ),
            text=risks,
            textposition="top center"
        ))
        
        # Add quadrants
        fig.add_shape(
            type="rect",
            x0=0, y0=0.5, x1=0.5, y1=1,
            line=dict(color="gray", width=1),
            fillcolor="rgba(255,0,0,0.1)"
        )
        fig.add_shape(
            type="rect",
            x0=0.5, y0=0.5, x1=1, y1=1,
            line=dict(color="gray", width=1),
            fillcolor="rgba(255,165,0,0.1)"
        )
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=0.5, y1=0.5,
            line=dict(color="gray", width=1),
            fillcolor="rgba(0,255,0,0.1)"
        )
        fig.add_shape(
            type="rect",
            x0=0.5, y0=0, x1=1, y1=0.5,
            line=dict(color="gray", width=1),
            fillcolor="rgba(255,255,0,0.1)"
        )
        
        # Update layout
        fig.update_layout(
            title='Risk Analysis Matrix',
            xaxis_title='Likelihood',
            yaxis_title='Impact',
            template='plotly_white',
            showlegend=False,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'risk_analysis.html')
        fig.write_image(self.output_dir / 'risk_analysis.png')

if __name__ == "__main__":
    generator = VisualizationGenerator()
    generator.generate_all_visualizations() 