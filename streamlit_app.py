"""
Real Estate Price Prediction - Streamlit Deployment App

This app uses pre-trained models to predict property prices and provide insights.
Models are loaded from the local 'models/' directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """
    Load all trained models from the models/ directory.
    Uses caching to avoid reloading on every interaction.
    """
    models_dir = Path("models")
    
    try:
        regressor = joblib.load(models_dir / "regressor.pkl")
        classifier = joblib.load(models_dir / "classifier.pkl")
        kmeans = joblib.load(models_dir / "kmeans.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        
        return regressor, classifier, kmeans, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.error("Please ensure all model files are in the 'models/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


def explain_prediction(features_dict, predicted_price, cluster_id, model_metadata=None):
    """
    Generate natural language explanation for the prediction.
    
    Args:
        features_dict: Dictionary of input features
        predicted_price: Predicted property price
        cluster_id: Assigned cluster/market segment
        model_metadata: Optional metadata for context
    
    Returns:
        str: Natural language explanation
    """
    explanation_parts = []
    
    # Main prediction
    explanation_parts.append(f"**Predicted Value:** ${predicted_price:,.2f}")
    
    # Property characteristics
    bedrooms = features_dict.get('bedrooms', 0)
    bathrooms = features_dict.get('bathrooms', 0)
    sqft_living = features_dict.get('sqft_living', 0)
    grade = features_dict.get('grade', 0)
    
    explanation_parts.append(f"\n**Property Details:**")
    explanation_parts.append(f"- {bedrooms} bedrooms, {bathrooms} bathrooms")
    explanation_parts.append(f"- {sqft_living:,.0f} sqft living area")
    if grade:
        explanation_parts.append(f"- Grade: {grade}")
    
    # Cluster interpretation
    cluster_names = {
        0: "Budget Segment",
        1: "Mid-Market Segment", 
        2: "Premium Segment",
        3: "Luxury Segment"
    }
    cluster_name = cluster_names.get(cluster_id, f"Segment {cluster_id}")
    explanation_parts.append(f"\n**Market Segment:** {cluster_name} (Cluster {cluster_id})")
    
    # Price per square foot insight
    if sqft_living > 0:
        price_per_sqft = predicted_price / sqft_living
        explanation_parts.append(f"- Price per sqft: ${price_per_sqft:,.2f}")
        
        # Market comparison (simplified thresholds)
        if price_per_sqft < 200:
            market_status = "below average"
        elif price_per_sqft < 400:
            market_status = "average"
        else:
            market_status = "above average"
        explanation_parts.append(f"- Market position: {market_status}")
    
    # Investment insight
    explanation_parts.append(f"\n**Investment Insight:**")
    if bedrooms >= 3 and bathrooms >= 2:
        explanation_parts.append("This property offers family-friendly space, which typically maintains strong resale value.")
    if sqft_living > 2000:
        explanation_parts.append("Large living area suggests premium positioning in the market.")
    
    explanation_parts.append("Consider location, neighborhood trends, and property condition for final investment decision.")
    
    return "\n".join(explanation_parts)


def prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode=None):
    """
    Prepare feature vector for model prediction.
    Creates all engineered features expected by the models.
    
    Args:
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        sqft_living: Living area in square feet
        grade: Property grade
        zipcode: Optional zipcode
    
    Returns:
        dict: Feature dictionary
    """
    features = {
        'bedrooms': float(bedrooms),
        'bathrooms': float(bathrooms),
        'sqft_living': float(sqft_living),
        'grade': float(grade) if grade else 7.0,  # Default grade if not provided
    }
    
    # Calculate derived features (matching notebook feature engineering)
    features['total_rooms'] = features['bedrooms'] + features['bathrooms']
    
    # Add zipcode if provided (encoded as numeric for simplicity)
    if zipcode:
        try:
            # Simple encoding: use last 3 digits as numeric feature
            zipcode_str = str(zipcode).strip()
            if zipcode_str:
                features['zipcode_encoded'] = float(zipcode_str[-3:]) / 1000.0
            else:
                features['zipcode_encoded'] = 0.0
        except:
            features['zipcode_encoded'] = 0.0
    
    # Add default values for other common features (set to 0 if not provided)
    # These won't affect prediction much if model doesn't use them
    default_features = {
        'sqft_lot': 0.0,
        'floors': 1.0,
        'waterfront': 0.0,
        'view': 0.0,
        'condition': 5.0,
        'sqft_above': features['sqft_living'],
        'sqft_basement': 0.0,
        'yr_built': 2000.0,
        'yr_renovated': 0.0,
        'lat': 47.5,  # Default to Seattle area
        'long': -122.3,
        'sqft_living15': features['sqft_living'],
        'sqft_lot15': 0.0,
    }
    
    # Only add defaults that might be in the model
    # (Tree models handle missing features gracefully)
    for key, default_val in default_features.items():
        if key not in features:
            features[key] = default_val
    
    return features


def predict_cluster(features_dict, scaler, kmeans):
    """
    Predict market segment cluster using KMeans.
    
    Args:
        features_dict: Input features
        scaler: Fitted StandardScaler
        kmeans: Fitted KMeans model
    
    Returns:
        int: Cluster ID
    """
    # Prepare clustering features (matching notebook logic)
    # The notebook uses: area, bedrooms, bathrooms, price, and other numeric features
    # For prediction, we use available features (price not available yet)
    cluster_features = []
    
    # Priority order: match what was used in training
    priority_features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms']
    
    # Try to infer expected features from scaler (if it has feature names)
    try:
        if hasattr(scaler, 'feature_names_in_'):
            cluster_features = list(scaler.feature_names_in_)
        else:
            # Use priority features that exist in our input
            cluster_features = [f for f in priority_features if f in features_dict]
    except:
        # Fallback: use common features
        cluster_features = [f for f in priority_features if f in features_dict]
    
    # Ensure we have at least some features
    if not cluster_features:
        cluster_features = ['bedrooms', 'bathrooms', 'sqft_living']
    
    # Create feature vector
    feature_vector = np.array([[features_dict.get(f, 0) for f in cluster_features]])
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Predict cluster
    cluster = kmeans.predict(feature_vector_scaled)[0]
    
    return cluster


def predict_price(features_dict, cluster_id, regressor):
    """
    Predict property price using XGBoost regressor.
    
    Args:
        features_dict: Input features
        cluster_id: Predicted cluster
        regressor: Fitted regressor model
    
    Returns:
        float: Predicted price
    """
    # Try to get feature names from model (XGBoost stores feature names)
    try:
        if hasattr(regressor, 'feature_names_in_'):
            feature_names = list(regressor.feature_names_in_)
        elif hasattr(regressor, 'get_booster'):
            # XGBoost native API
            feature_names = regressor.get_booster().feature_names
            if feature_names is None:
                raise AttributeError
        else:
            raise AttributeError
    except (AttributeError, TypeError):
        # Fallback: use common feature names (adjust based on your model)
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms', 'cluster']
        # Add other common features that might exist
        possible_features = ['sqft_lot', 'floors', 'waterfront', 'view', 'condition', 
                            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                            'lat', 'long', 'sqft_living15', 'sqft_lot15']
        feature_names.extend([f for f in possible_features if f in features_dict])
    
    # Build feature vector matching model's expected features
    feature_vector = []
    for name in feature_names:
        if name == 'cluster':
            feature_vector.append(float(cluster_id))
        elif name in features_dict:
            feature_vector.append(float(features_dict[name]))
        else:
            # Fill missing features with 0 (tree models handle this)
            feature_vector.append(0.0)
    
    feature_vector = np.array([feature_vector])
    
    # Predict price
    predicted_price = regressor.predict(feature_vector)[0]
    
    # Ensure non-negative price
    predicted_price = max(0, predicted_price)
    
    return predicted_price


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🏠 Real Estate Price Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models (cached)
    with st.spinner("Loading models..."):
        regressor, classifier, kmeans, scaler = load_models()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Property Details")
        st.markdown("Enter property characteristics to get price prediction.")
        
        # Input form
        with st.form("prediction_form"):
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
                help="Number of bedrooms"
            )
            
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Number of bathrooms"
            )
            
            sqft_living = st.number_input(
                "Living Area (sqft)",
                min_value=0,
                max_value=20000,
                value=2000,
                step=100,
                help="Total living area in square feet"
            )
            
            grade = st.number_input(
                "Property Grade",
                min_value=1,
                max_value=13,
                value=7,
                step=1,
                help="Overall property grade (1-13, higher is better)"
            )
            
            zipcode = st.text_input(
                "Zipcode (Optional)",
                value="",
                help="Property zipcode (optional)"
            )
            
            submitted = st.form_submit_button(
                "🔮 Predict Price",
                use_container_width=True,
                type="primary"
            )
    
    # Main content area
    if submitted:
        # Prepare features
        features_dict = prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode if zipcode else None)
        
        # Prediction pipeline
        with st.spinner("Analyzing property..."):
            # Step 1: Predict cluster
            cluster_id = predict_cluster(features_dict, scaler, kmeans)
            
            # Step 2: Predict price
            predicted_price = predict_price(features_dict, cluster_id, regressor)
            
            # Step 3: Generate explanation
            explanation = explain_prediction(features_dict, predicted_price, cluster_id)
        
        # Display results
        st.markdown("## 📊 Prediction Results")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Price",
                f"${predicted_price:,.0f}",
                help="Model's price prediction"
            )
        
        with col2:
            cluster_names = {
                0: "Budget",
                1: "Mid-Market",
                2: "Premium", 
                3: "Luxury"
            }
            st.metric(
                "Market Segment",
                cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                help="Property market segment"
            )
        
        with col3:
            if sqft_living > 0:
                price_per_sqft = predicted_price / sqft_living
                st.metric(
                    "Price per SqFt",
                    f"${price_per_sqft:,.2f}",
                    help="Price per square foot"
                )
        
        # Explanation box
        st.markdown("---")
        st.markdown("## 💡 Prediction Explanation")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(explanation)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("---")
        with st.expander("📈 Additional Insights"):
            st.info("""
            **Model Information:**
            - This prediction uses XGBoost regression model trained on historical property data
            - Market segmentation via KMeans clustering identifies property tier
            - Predictions are estimates based on property characteristics
            
            **Disclaimer:**
            - Actual property values may vary based on location, condition, market conditions, and other factors
            - This tool is for estimation purposes only
            - Consult with real estate professionals for accurate valuations
            """)
    
    else:
        # Welcome message
        st.info("👈 **Get started:** Fill out the property details in the sidebar and click 'Predict Price' to see the prediction.")
        
        # Example
        st.markdown("### 📝 Example Input")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("""
            **Typical Family Home:**
            - Bedrooms: 3
            - Bathrooms: 2.5
            - Living Area: 2,000 sqft
            - Grade: 7
            """)
        
        with example_col2:
            st.markdown("""
            **Luxury Property:**
            - Bedrooms: 4
            - Bathrooms: 3.5
            - Living Area: 3,500 sqft
            - Grade: 10
            """)


if __name__ == "__main__":
    main()

