import os
import sys
from src.dashboard.app import app

if __name__ == '__main__':
    # Création du répertoire models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Lancement du tableau de bord
    print("Lancement du tableau de bord d'analyse prédictive...")
    print("Accédez à l'interface web à l'adresse: http://localhost:8050")
    app.run_server(debug=True, port=8050) 