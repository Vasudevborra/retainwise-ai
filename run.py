"""
Startup and Execution Orchestrator for Customer Intelligence System
1. Verifies dataset presence (or generates it automatically)
2. Runs the Feature Engineering Pipeline
3. Trains the K-Means Behavioral Clustering model
4. Starts the local Flask server on port 5000
"""

import os
import sys
import webbrowser

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_generator import generate_dataset
from src.feature_pipeline import build_feature_store
from src.clustering_engine import run_clustering

def initialize_system():
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    customers_csv = os.path.join(data_dir, "customers.csv")
    features_csv = os.path.join(data_dir, "customer_features.csv")
    clustered_csv = os.path.join(data_dir, "clustered_customers.csv")
    
    # 1. Step 1: Data Generation if needed
    if not os.path.exists(customers_csv):
        print("Step 1/3: No transaction data detected. Generating synthetic realistic datasets...")
        generate_dataset(n_customers=2000, output_dir=data_dir)
    else:
        print("Step 1/3: Verified raw customer and transaction tables.")
        
    # 2. Step 2: Feature Store compilation
    if not os.path.exists(features_csv):
        print("Step 2/3: Compiling customer-level behavioral feature store...")
        build_feature_store(data_dir=data_dir)
    else:
        print("Step 2/3: Verified customer feature store.")
        
    # 3. Step 3: Unsupervised Machine Learning Clustering
    if not os.path.exists(clustered_csv):
        print("Step 3/3: Running K-Means clustering algorithm...")
        run_clustering(data_dir=data_dir)
    else:
        print("Step 3/3: Verified behavioral cluster assignments.")

if __name__ == "__main__":
    print("\n==================================================================")
    print("      ANTIGRAVITY CUSTOMER INTELLIGENCE & CLUSTERING ENGINE       ")
    print("==================================================================\n")
    
    initialize_system()
    
    from src.server import app
    print("\nStarting local Customer Intelligence Studio web server...")
    print("Serving dashboard at: http://127.0.0.1:5000\n")
    
    app.run(host="127.0.0.1", port=5000, debug=False)
