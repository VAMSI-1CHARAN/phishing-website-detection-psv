import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import tldextract
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urlparse
import whois
import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Define PyTorch models
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_features(url):
    try:
        features = []
        parsed_url = urlparse(url)
        domain_info = tldextract.extract(url)
        domain = domain_info.domain + '.' + domain_info.suffix
        
        # Basic features
        features.append(len(url))
        features.append(url.count('.'))
        features.append(url.count('-'))
        features.append(1 if parsed_url.scheme == 'https' else 0)
        features.append(1 if '-' in domain_info.domain else 0)
        features.append(1 if re.search(r'[^a-zA-Z0-9-.]', domain_info.domain) else 0)
        features.append(1 if any(short in url for short in ['bit.ly', 'tinyurl', 'goo.gl']) else 0)
        features.append(url.count('/'))
        features.append(1)  # Web traffic placeholder
        
        # WHOIS features
        domain_age = 0
        dns_record = 0
        try:
            whois_info = whois.whois(domain)
            if whois_info.domain_name:
                dns_record = 1
                creation_date = whois_info.creation_date
                if creation_date:
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]
                    domain_age = (datetime.datetime.now() - creation_date).days / 365
        except:
            pass
            
        features.append(dns_record)
        features.append(1)  # Google index placeholder
        features.append(1)  # Page rank placeholder
        features.append(domain_age)
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise

def create_training_data():
    legitimate_urls = [
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.amazon.com",
        "https://www.apple.com",
        "https://www.microsoft.com"
    ]
    
    phishing_urls = [
        "http://googlle-secure.com",
        "http://facebook-login.com",
        "http://paypal-secure.com",
        "http://apple-icloud.com",
        "http://microsoft-update.com"
    ]
    
    X = []
    y = []
    
    for url in legitimate_urls:
        try:
            features = extract_features(url)
            X.append(features[0])
            y.append(0)
        except Exception as e:
            logger.error(f"Error processing legitimate URL {url}: {str(e)}")
    
    for url in phishing_urls:
        try:
            features = extract_features(url)
            X.append(features[0])
            y.append(1)
        except Exception as e:
            logger.error(f"Error processing phishing URL {url}: {str(e)}")
    
    return np.array(X), np.array(y)

# Initialize Flask app
app = Flask(__name__)

# Global model variable
model = None

def initialize_model():
    global model
    try:
        # Create and train model
        X, y = create_training_data()
        input_size = X.shape[1]
        
        model = CNNModel(input_size)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Train the model
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        logger.info("Model initialized and trained successfully")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({
                'error': 'Please enter a URL'
            })
        
        if not is_valid_url(url):
            return jsonify({
                'error': 'Please enter a valid URL (e.g., https://www.example.com)'
            })
        
        # Extract features
        features = extract_features(url)
        features_tensor = torch.FloatTensor(features)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(features_tensor)
            confidence = float(prediction.item())
            
            if confidence > 0.5:
                result = "Phishing"
            else:
                result = "Legitimate"
            
            return jsonify({
                'url': url,
                'prediction': result,
                'confidence': f"{confidence:.2%}"
            })
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Error processing URL: {str(e)}'
        })

if __name__ == '__main__':
    # Initialize the model
    initialize_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8081)
