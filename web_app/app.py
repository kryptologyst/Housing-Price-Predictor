"""
Streamlit Web Interface for Housing Price Predictor.

This module provides a modern web interface for the housing price predictor
using Streamlit, allowing users to interact with the models through a web browser.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from housing_predictor import HousingPredictor, DataLoader, VisualizationUtils
from housing_predictor.config.settings import config


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Housing Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        dataset = st.selectbox(
            "Select Dataset",
            ["California Housing", "Synthetic Data", "Zillow Data"],
            help="Choose the dataset to use for training and prediction"
        )
        
        # Model selection
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["Linear Regression", "Random Forest", "Gradient Boosting", "Neural Network", "XGBoost"],
            default=["Linear Regression", "Random Forest"],
            help="Choose which models to train and compare"
        )
        
        # Training parameters
        st.subheader("📊 Training Parameters")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 1000, 42)
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        remove_outliers = st.checkbox("Remove Outliers", False)
        scale_features = st.checkbox("Scale Features", True)
        feature_selection = st.checkbox("Feature Selection", False)
        
        if feature_selection:
            n_features = st.slider("Number of Features", 5, 20, 10)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Data Overview", "🤖 Model Training", "📈 Predictions", "📊 Analysis"])
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = HousingPredictor()
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        if st.button("Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                try:
                    # Load dataset based on selection
                    if dataset == "California Housing":
                        X, y = st.session_state.predictor.load_dataset('california')
                        dataset_name = "California Housing"
                    elif dataset == "Synthetic Data":
                        X, y = st.session_state.predictor.load_dataset('synthetic', n_samples=1000)
                        dataset_name = "Synthetic Housing Data"
                    else:  # Zillow Data
                        X, y = st.session_state.predictor.load_dataset('zillow')
                        dataset_name = "Zillow Housing Data"
                    
                    # Store in session state
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.dataset_name = dataset_name
                    
                    st.success(f"✅ {dataset_name} loaded successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
        
        # Display dataset info
        if 'X' in st.session_state:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", f"{st.session_state.X.shape[0]:,}")
            with col2:
                st.metric("Features", st.session_state.X.shape[1])
            with col3:
                st.metric("Avg Price", f"${st.session_state.y.mean():,.0f}")
            with col4:
                st.metric("Price Range", f"${st.session_state.y.min():,.0f} - ${st.session_state.y.max():,.0f}")
            
            # Data visualization
            st.subheader("📈 Data Visualization")
            
            # Create sample data for visualization
            if st.session_state.X.shape[1] <= 8:  # California housing
                feature_names = [
                    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude'
                ]
            else:
                feature_names = [f"Feature_{i}" for i in range(st.session_state.X.shape[1])]
            
            df_viz = pd.DataFrame(st.session_state.X, columns=feature_names)
            df_viz['Price'] = st.session_state.y
            
            # Price distribution
            fig1 = px.histogram(df_viz, x='Price', nbins=50, title='Price Distribution')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Feature correlation
            if len(feature_names) > 1:
                corr_matrix = df_viz.corr()
                fig2 = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                               title='Feature Correlation Matrix')
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("🤖 Model Training")
        
        if 'X' not in st.session_state:
            st.warning("⚠️ Please load a dataset first in the Data Overview tab.")
        else:
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        # Preprocess data
                        X_processed, y_processed = st.session_state.predictor.preprocess_data(
                            st.session_state.X, st.session_state.y, st.session_state.dataset_name
                        )
                        
                        # Split data
                        st.session_state.predictor.split_data(
                            X_processed, y_processed, test_size=test_size, random_state=random_state
                        )
                        
                        # Train models
                        st.session_state.predictor.train_models(models_to_train)
                        
                        # Evaluate models
                        evaluation_results = st.session_state.predictor.evaluate_models()
                        st.session_state.evaluation_results = evaluation_results
                        
                        st.success("✅ Models trained successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error training models: {str(e)}")
            
            # Display results
            if 'evaluation_results' in st.session_state:
                st.subheader("📊 Model Performance")
                
                # Create performance comparison
                results_df = pd.DataFrame(st.session_state.evaluation_results).T
                results_df = results_df.round(4)
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    st.dataframe(results_df, use_container_width=True)
                
                with col2:
                    st.subheader("Best Model")
                    best_model = results_df['R2'].idxmax()
                    best_r2 = results_df.loc[best_model, 'R2']
                    
                    st.metric("Model", best_model)
                    st.metric("R² Score", f"{best_r2:.4f}")
                    st.metric("RMSE", f"${results_df.loc[best_model, 'RMSE']:,.0f}")
                
                # Performance comparison chart
                fig = go.Figure()
                
                models = results_df.index
                metrics = ['RMSE', 'MAE', 'R2']
                
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=models,
                        y=results_df[metric],
                        text=results_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔮 Make Predictions")
        
        if 'evaluation_results' not in st.session_state:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            st.subheader("📝 Input Features")
            
            # Get feature names
            if hasattr(st.session_state.predictor, 'feature_names'):
                feature_names = st.session_state.predictor.feature_names
            else:
                feature_names = [f"Feature_{i}" for i in range(st.session_state.X.shape[1])]
            
            # Create input form
            input_values = {}
            cols = st.columns(3)
            
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    # Get reasonable default value
                    if i < st.session_state.X.shape[0]:
                        default_val = float(st.session_state.X[0, i])
                    else:
                        default_val = 0.0
                    
                    input_values[feature] = st.number_input(
                        feature,
                        value=default_val,
                        format="%.2f"
                    )
            
            # Model selection for prediction
            model_for_prediction = st.selectbox(
                "Select Model for Prediction",
                list(st.session_state.evaluation_results.keys()),
                help="Choose which trained model to use for prediction"
            )
            
            if st.button("Predict Price", type="primary"):
                try:
                    # Prepare input data
                    input_array = np.array([list(input_values.values())]).reshape(1, -1)
                    
                    # Make prediction
                    prediction = st.session_state.predictor.predict(input_array, model_for_prediction)
                    
                    # Display result
                    st.success(f"🏠 Predicted Price: **${prediction[0]:,.2f}**")
                    
                    # Show confidence (based on R² score)
                    r2_score = st.session_state.evaluation_results[model_for_prediction]['R2']
                    confidence = min(100, max(0, r2_score * 100))
                    
                    st.info(f"Model Confidence: {confidence:.1f}% (R² = {r2_score:.3f})")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    with tab4:
        st.header("📊 Advanced Analysis")
        
        if 'evaluation_results' not in st.session_state:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            # Feature importance
            st.subheader("🎯 Feature Importance")
            
            if st.button("Show Feature Importance"):
                try:
                    importance = st.session_state.predictor.get_feature_importance()
                    
                    if importance:
                        # Create feature importance plot
                        features = list(importance.keys())
                        values = list(importance.values())
                        
                        fig = px.bar(
                            x=values, y=features,
                            orientation='h',
                            title="Feature Importance",
                            labels={'x': 'Importance Score', 'y': 'Features'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature importance not available for selected model.")
                        
                except Exception as e:
                    st.error(f"❌ Error showing feature importance: {str(e)}")
            
            # Model comparison
            st.subheader("📈 Model Comparison")
            
            if st.button("Show Detailed Comparison"):
                try:
                    # Actual vs Predicted plot
                    X_test, y_test = st.session_state.predictor.test_data
                    y_pred = st.session_state.predictor.predict(X_test)
                    
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        title="Actual vs Predicted Values",
                        labels={'x': 'Actual Price', 'y': 'Predicted Price'}
                    )
                    
                    # Add perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error showing comparison: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        🏠 Housing Price Predictor v2.0 | Built with Streamlit & Python
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
