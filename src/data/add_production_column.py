import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'

# Charger le fichier existant
file_path = RAW_DATA_DIR / 'BrentOilPrices.csv'
df = pd.read_csv(file_path)

# Générer une colonne de production synthétique (en milliers de barils/jour)
np.random.seed(42)
# Production réaliste : entre 800 et 1200 milliers de barils/jour
production = np.random.normal(loc=1000, scale=80, size=len(df))
production = np.clip(production, 800, 1200)
df['production'] = production.round(0)

# Harmoniser le nom de la colonne date
if 'Date' in df.columns:
    df.rename(columns={'Date': 'date'}, inplace=True)
if 'Price' in df.columns:
    df.rename(columns={'Price': 'price'}, inplace=True)

# Sauvegarder le nouveau fichier
output_path = RAW_DATA_DIR / 'BrentOilPrices_with_production.csv'
df.to_csv(output_path, index=False)
print(f"Fichier enrichi sauvegardé : {output_path}") 