import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📊 Data Analysis Dashboard")

# File uploader for dataset
st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Sidebar filters
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if "Age" in numeric_columns:
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
        df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1])]
    
    # Create card layout for data summary, feature distribution, and correlation heatmap
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Data Summary"):
            st.write(df.describe())
    
    with col2:
        if st.button("📊 Feature Distributions"):
            feature = st.selectbox("Select feature", numeric_columns)
            fig = px.histogram(df, x=feature, barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if st.button("📉 Correlation Heatmap"):
            if len(numeric_columns) > 1:
                corr = df[numeric_columns].corr()
                fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Not enough numeric features for correlation heatmap.")
    
    # Regression Section
    st.sidebar.header("📊 Regression Model")
    feature_selection = st.sidebar.multiselect("Select Features for Regression", options=numeric_columns)
    target_selection = st.sidebar.selectbox("Select Target Variable", options=numeric_columns)
    
    if st.sidebar.button("Train Regression Model"):
        if feature_selection and target_selection:
            X = df[feature_selection]
            y = df[target_selection]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            st.write(model.summary())
            
            regression_result_df = pd.DataFrame({"Actual": y, "Predicted": model.predict(X)})
            fig = px.scatter(regression_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Values")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Please select both features and target variable for regression.")
    
    # Machine Learning Model - Logistic Regression
    st.sidebar.header("🧠 Classification Model")
    if st.sidebar.button("Train Classification Model"):
        X = df[numeric_columns[:-1]]
        y = df[numeric_columns[-1]].astype(int) if numeric_columns else None
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            joblib.dump(model, "model.pkl")
            st.sidebar.success("Model trained successfully!")
        else:
            st.sidebar.write("No valid target variable found for classification.")
    
    # Download option
    st.sidebar.download_button("Download Processed Data", df.to_csv(index=False), "processed_data.csv", "text/csv")
else:
    st.write("Please upload a CSV file to start.")
