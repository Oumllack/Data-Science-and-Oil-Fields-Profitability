"""
Analyse de rentabilité des champs pétroliers.
Ce script effectue l'analyse complète des données de production et de prix du pétrole.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuration des chemins
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Configuration de l'affichage
plt.style.use('default')  # Utilisation du style par défaut
sns.set_theme()  # Utilisation du thème par défaut de seaborn
pd.set_option('display.max_columns', None)

def load_data():
    """Charge les données depuis le dossier raw_data"""
    csv_files = list(RAW_DATA_DIR.glob('*.csv'))
    
    if not csv_files:
        # print("Veuillez télécharger le dataset Kaggle d'abord.")
        raise FileNotFoundError("Aucun fichier CSV trouvé dans le dossier raw_data. "
                               "Veuillez télécharger le dataset Kaggle d'abord.")
    
    data = {}
    for file in csv_files:
        name = file.stem
        data[name] = pd.read_csv(file)
        print(f"\nChargé {name}:")
        print(f"Dimensions: {data[name].shape}")
        print("\nAperçu des données:")
        print(data[name].head())
        print("\nInformations sur les colonnes:")
        print(data[name].info())
    
    return data

def analyze_data(df):
    """Analyse exploratoire des données"""
    print("\n=== Analyse Exploratoire ===")
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    print(df.describe())
    
    # Valeurs manquantes
    print("\nValeurs manquantes:")
    print(df.isnull().sum())
    
    # Visualisations
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        # Graphique des prix du pétrole
        if 'price' in df.columns:
            fig = px.line(df, x='date', y='price', 
                         title='Évolution des Prix du Pétrole')
            fig.write_html(PROCESSED_DATA_DIR / 'prix_petrole.html')
        
        # Graphique de la production
        if 'production' in df.columns:
            fig = px.line(df, x='date', y='production',
                         title='Évolution de la Production')
            fig.write_html(PROCESSED_DATA_DIR / 'production.html')
    
    # Matrice de corrélation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix,
                       title='Matrice de Corrélation',
                       color_continuous_scale='RdBu')
        fig.write_html(PROCESSED_DATA_DIR / 'correlation.html')

def calculate_profitability(df):
    """Calcule les métriques de rentabilité"""
    print("\n=== Calcul de la Rentabilité ===")
    
    df_profit = df.copy()
    
    # Vérification des colonnes nécessaires
    required_cols = ['price', 'production']
    if not all(col in df.columns for col in required_cols):
        print(f"Colonnes manquantes. Nécessaire: {required_cols}")
        return None
    
    # Calcul des revenus
    df_profit['revenue'] = df_profit['price'] * df_profit['production']
    
    # Estimation des coûts d'exploitation
    if 'operating_cost' not in df.columns:
        avg_cost_per_barrel = 30  # USD/baril
        df_profit['operating_cost'] = df_profit['production'] * avg_cost_per_barrel
    
    # Calcul de la marge brute
    df_profit['gross_margin'] = df_profit['revenue'] - df_profit['operating_cost']
    
    # Calcul du ROI
    df_profit['roi'] = df_profit['gross_margin'] / df_profit['operating_cost']
    
    # Sauvegarde des résultats
    df_profit.to_csv(PROCESSED_DATA_DIR / 'rentabilite.csv', index=False)
    
    # Visualisations
    if 'date' in df.columns:
        # Graphique des métriques de rentabilité
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_profit['date'], y=df_profit['revenue'],
                                name='Revenus', mode='lines'))
        fig.add_trace(go.Scatter(x=df_profit['date'], y=df_profit['operating_cost'],
                                name='Coûts', mode='lines'))
        fig.add_trace(go.Scatter(x=df_profit['date'], y=df_profit['gross_margin'],
                                name='Marge Brute', mode='lines'))
        fig.update_layout(title='Évolution des Métriques de Rentabilité',
                         xaxis_title='Date',
                         yaxis_title='USD')
        fig.write_html(PROCESSED_DATA_DIR / 'metriques_rentabilite.html')
        
        # Graphique du ROI
        fig = px.line(df_profit, x='date', y='roi',
                     title='Évolution du ROI')
        fig.write_html(PROCESSED_DATA_DIR / 'roi.html')
    
    return df_profit

def sensitivity_analysis(df, base_price, base_production, base_cost):
    """Analyse de sensibilité des paramètres clés"""
    print("\n=== Analyse de Sensibilité ===")
    
    variations = np.linspace(0.8, 1.2, 5)
    results = []
    
    # Test des variations de prix
    for var in variations:
        price = base_price * var
        revenue = price * base_production
        gross_margin = revenue - base_cost
        roi = gross_margin / base_cost
        results.append({
            'paramètre': 'Prix',
            'variation': f"{(var-1)*100:.0f}%",
            'roi': roi
        })
    
    # Test des variations de production
    for var in variations:
        production = base_production * var
        revenue = base_price * production
        gross_margin = revenue - base_cost
        roi = gross_margin / base_cost
        results.append({
            'paramètre': 'Production',
            'variation': f"{(var-1)*100:.0f}%",
            'roi': roi
        })
    
    # Test des variations de coûts
    for var in variations:
        cost = base_cost * var
        revenue = base_price * base_production
        gross_margin = revenue - cost
        roi = gross_margin / cost
        results.append({
            'paramètre': 'Coûts',
            'variation': f"{(var-1)*100:.0f}%",
            'roi': roi
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DATA_DIR / 'analyse_sensibilite.csv', index=False)
    
    # Visualisation
    fig = px.line(results_df, x='variation', y='roi', color='paramètre',
                  title='Analyse de Sensibilité du ROI',
                  labels={'variation': 'Variation du Paramètre',
                         'roi': 'ROI'})
    fig.write_html(PROCESSED_DATA_DIR / 'sensibilite.html')
    
    return results_df

def regression_and_prediction(df):
    """Réalise des régressions linéaires et des prévisions simples"""
    print("\n=== Régression et Prédiction ===")
    results = {}
    # Nettoyage et préparation
    df = df.copy()
    df = df.dropna(subset=['price', 'production', 'roi'])
    # Régression linéaire : prix ~ production
    X = df[['production']]
    y = df['price']
    model_price = LinearRegression()
    model_price.fit(X, y)
    score_price = model_price.score(X, y)
    print(f"R² (prix ~ production) : {score_price:.3f}")
    print(f"Prix = {model_price.coef_[0]:.3f} * Production + {model_price.intercept_:.3f}")
    results['model_price'] = model_price
    # Régression linéaire : ROI ~ production
    X_roi = df[['production']]
    y_roi = df['roi']
    model_roi = LinearRegression()
    model_roi.fit(X_roi, y_roi)
    score_roi = model_roi.score(X_roi, y_roi)
    print(f"R² (ROI ~ production) : {score_roi:.3f}")
    print(f"ROI = {model_roi.coef_[0]:.6f} * Production + {model_roi.intercept_:.3f}")
    results['model_roi'] = model_roi
    # Prédiction sur les 30 prochains jours (projection simple)
    last_date = pd.to_datetime(df['date']).max()
    future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]
    # On suppose que la production reste stable autour de la moyenne
    future_production = np.full(len(future_dates), df['production'].mean())
    # Prédiction du prix
    predicted_prices = model_price.predict(future_production.reshape(-1, 1))
    # Prédiction du ROI
    predicted_roi = model_roi.predict(future_production.reshape(-1, 1))
    # Résumé
    future_df = pd.DataFrame({
        'date': future_dates,
        'production': future_production,
        'predicted_price': predicted_prices,
        'predicted_roi': predicted_roi
    })
    print("\nPrévisions sur 30 jours :")
    print(future_df.head())
    # Sauvegarde
    future_df.to_csv(PROCESSED_DATA_DIR / 'previsions_30j.csv', index=False)
    # Visualisation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['price'], name='Prix historique', mode='lines'))
    fig.add_trace(go.Scatter(x=future_df['date'], y=future_df['predicted_price'], name='Prix prédit', mode='lines'))
    fig.update_layout(title='Prévision du Prix du Pétrole (30 jours)', xaxis_title='Date', yaxis_title='Prix')
    fig.write_html(PROCESSED_DATA_DIR / 'prevision_prix.html')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['roi'], name='ROI historique', mode='lines'))
    fig.add_trace(go.Scatter(x=future_df['date'], y=future_df['predicted_roi'], name='ROI prédit', mode='lines'))
    fig.update_layout(title='Prévision du ROI (30 jours)', xaxis_title='Date', yaxis_title='ROI')
    fig.write_html(PROCESSED_DATA_DIR / 'prevision_roi.html')
    return results, future_df

def main():
    """Fonction principale"""
    print("=== Analyse de Rentabilité des Champs Pétroliers ===")
    
    try:
        # Création des répertoires si nécessaire
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Chargement des données
        data = load_data()
        
        # Analyse pour chaque dataset
        for name, df in data.items():
            print(f"\n=== Analyse de {name} ===")
            
            # Analyse exploratoire
            analyze_data(df)
            
            # Calcul de la rentabilité
            if 'price' in df.columns and 'production' in df.columns:
                df_profit = calculate_profitability(df)
                
                if df_profit is not None:
                    # Analyse de sensibilité
                    base_price = df['price'].mean()
                    base_production = df['production'].mean()
                    base_cost = df_profit['operating_cost'].mean()
                    
                    sensitivity_results = sensitivity_analysis(
                        df_profit, base_price, base_production, base_cost
                    )
                    print("\nRésultats de l'analyse de sensibilité:")
                    print(sensitivity_results)
                    # Régression et prévision
                    regression_and_prediction(df_profit)
        
        print("\n=== Analyse terminée ===")
        print(f"Les résultats ont été sauvegardés dans {PROCESSED_DATA_DIR}")
        
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {e}")
        raise

if __name__ == "__main__":
    main() 