from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import gc

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Custom JSON encoder to handle NaN values
def handle_nans(obj):
    if isinstance(obj, (float, np.floating)):
        return None if pd.isna(obj) else obj
    elif isinstance(obj, (list, tuple)):
        return [handle_nans(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: handle_nans(v) for k, v in obj.items()}
    return obj

# Load the dataset and ML model
try:
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), 'kindle_data-v2.csv')
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path).fillna('')
    
    # Create content for ML (using available columns)
    content_cols = [col for col in ['title', 'author', 'category_name'] if col in df.columns]
    df['content'] = df[content_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    # Initialize ML components
    model_path = os.path.join(os.path.dirname(__file__), 'book_recommender.pkl')
    tfidf_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        print("Loading pre-trained ML model...")
        nn_model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
    else:
        print("Training ML model for the first time...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = tfidf.fit_transform(df['content'])
        
        nn_model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute', n_jobs=-1)
        nn_model.fit(tfidf_matrix)
        
        joblib.dump(nn_model, model_path)
        joblib.dump(tfidf, tfidf_path)
    
    print(f"Successfully loaded {len(df)} records and ML model")
    
except Exception as e:
    print(f"Initialization Error: {str(e)}")
    df = pd.DataFrame()
    nn_model = None
    tfidf = None

# Existing routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/books')
def get_books():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        return jsonify({
            "books": handle_nans(df.to_dict('records')),
            "total": len(df)
        })
    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# New ML recommendation endpoint
@app.route('/api/recommend', methods=['GET'])
def recommend():
    if df.empty or nn_model is None:
        return jsonify({"error": "Recommendation system not ready"}), 503
    
    try:
        title = request.args.get('title', '').strip()
        if not title:
            return jsonify({"error": "Title parameter is required"}), 400
        
        # Find the book
        matches = df[df['title'].str.contains(title, case=False, regex=False)]
        if len(matches) == 0:
            return jsonify({"error": "No matching books found"}), 404
        
        # Get recommendations
        book_idx = matches.index[0]
        vector = tfidf.transform([df.iloc[book_idx]['content']])
        distances, indices = nn_model.kneighbors(vector)
        
        # Prepare results (exclude the queried book itself)
        recommendations = df.iloc[indices[0][1:11]].to_dict('records')
        
        return jsonify({
            "query": df.iloc[book_idx]['title'],
            "recommendations": handle_nans(recommendations)
        })
        
    except Exception as e:
        print(f"Recommendation Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run()