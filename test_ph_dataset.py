"""
Test script to validate advanced ML models with the provided pH dataset
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta

# Create the pH dataset from user's data
ph_data = [
    (0, 7.5), (1, 7.577645714), (2, 7.65), (3, 7.712132034), (4, 7.759807621),
    (5, 7.789777748), (6, 7.8), (7, 7.789777748), (8, 7.759807621), (9, 7.712132034),
    (10, 7.65), (11, 7.577645714), (12, 7.5), (13, 7.422354286), (14, 7.35),
    (15, 7.287867966), (16, 7.240192379), (17, 7.210222252), (18, 7.2), (19, 7.210222252),
    (20, 7.240192379), (21, 7.287867966), (22, 7.35), (23, 7.422354286), (24, 7.5),
    (25, 7.577645714), (26, 7.65), (27, 7.712132034), (28, 7.759807621), (29, 7.789777748),
    (30, 7.8), (31, 7.789777748), (32, 7.759807621), (33, 7.712132034), (34, 7.65),
    (35, 7.577645714), (36, 7.5), (37, 7.422354286), (38, 7.35), (39, 7.287867966),
    (40, 7.240192379), (41, 7.210222252), (42, 7.2), (43, 7.210222252), (44, 7.240192379),
    (45, 7.287867966), (46, 7.35), (47, 7.422354286), (48, 7.5)
]

# Create DataFrame
df = pd.DataFrame(ph_data, columns=['time_step', 'pH'])

# Add proper timestamp column
start_time = datetime(2025, 1, 1, 0, 0, 0)
df['timestamp'] = [start_time + timedelta(hours=i) for i in range(len(df))]

# Save to CSV
df.to_csv('/app/test_ph_data.csv', index=False)
print("✅ Created pH test dataset with shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Test the backend API
BASE_URL = "http://localhost:8001/api"

def test_advanced_ml_pipeline():
    print("\n🧪 Testing Advanced ML Pipeline with pH Data")
    
    # 1. Test file upload
    print("\n1. Testing file upload...")
    files = {'file': open('/app/test_ph_data.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/upload-data", files=files)
    files['file'].close()
    
    if response.status_code == 200:
        print("✅ File upload successful")
        upload_result = response.json()
        data_id = upload_result['data_id']
        print(f"Data ID: {data_id}")
    else:
        print(f"❌ File upload failed: {response.status_code}")
        print(response.text)
        return False
    
    # 2. Test data quality report
    print("\n2. Testing data quality report...")
    response = requests.get(f"{BASE_URL}/data-quality-report")
    if response.status_code == 200:
        print("✅ Data quality report successful")
        quality_report = response.json()
        print(f"Quality Score: {quality_report['quality_score']}")
    else:
        print(f"❌ Data quality report failed: {response.status_code}")
        print(response.text)
    
    # 3. Test advanced model training
    models_to_test = ['dlinear', 'lstm', 'lightgbm']
    
    for model_type in models_to_test:
        print(f"\n3. Testing {model_type} model training...")
        
        training_params = {
            'time_column': 'timestamp',
            'target_column': 'pH',
            'sequence_length': 10,
            'prediction_horizon': 5,
            'epochs': 20,
            'batch_size': 8,
            'learning_rate': 0.001
        }
        
        response = requests.post(f"{BASE_URL}/train-model", params={
            'data_id': data_id,
            'model_type': model_type
        }, json=training_params)
        
        if response.status_code == 200:
            print(f"✅ {model_type} model training successful")
            result = response.json()
            print(f"Model ID: {result.get('model_id', 'N/A')}")
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                print(f"RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"MAE: {metrics.get('mae', 'N/A')}")
                print(f"R²: {metrics.get('r2', 'N/A')}")
        else:
            print(f"❌ {model_type} model training failed: {response.status_code}")
            print(response.text)
    
    # 4. Test supported models
    print("\n4. Testing supported models endpoint...")
    response = requests.get(f"{BASE_URL}/supported-models")
    if response.status_code == 200:
        print("✅ Supported models endpoint successful")
        models = response.json()
        print(f"Supported models: {models}")
    else:
        print(f"❌ Supported models failed: {response.status_code}")
    
    # 5. Test model comparison
    print("\n5. Testing model comparison...")
    response = requests.get(f"{BASE_URL}/model-comparison", params={
        'time_column': 'timestamp',
        'target_column': 'pH'
    })
    
    if response.status_code == 200:
        print("✅ Model comparison successful")
        comparison = response.json()
        print("Comparison results:", comparison.get('comparison_results', 'N/A'))
    else:
        print(f"❌ Model comparison failed: {response.status_code}")
        print(response.text)
    
    # 6. Test hyperparameter optimization
    print("\n6. Testing hyperparameter optimization...")
    response = requests.post(f"{BASE_URL}/optimize-hyperparameters", json={
        'model_type': 'dlinear',
        'time_column': 'timestamp',
        'target_column': 'pH',
        'n_trials': 5
    })
    
    if response.status_code == 200:
        print("✅ Hyperparameter optimization successful")
        optimization = response.json()
        print("Best parameters:", optimization.get('best_parameters', 'N/A'))
    else:
        print(f"❌ Hyperparameter optimization failed: {response.status_code}")
        print(response.text)
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Advanced ML Testing with pH Dataset")
    test_advanced_ml_pipeline()
    print("\n✅ Testing completed!")