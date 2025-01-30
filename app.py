import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

# Streamlit app
st.set_page_config(page_title="Dynamic Regression Dashboard", layout="wide")
st.title("📊 Dynamic Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
columns = df.columns.tolist()

# Select numerical features
numerical_features = df.select_dtypes(include=['number']).columns.tolist()
if 'Age' in numerical_features:
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
    df = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]

# Create card layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 Data Summary"):
        st.write(df.describe())

with col2:
    if st.button("📊 Feature Distributions"):
        feature = st.selectbox("Select feature", numerical_features)
        fig = px.histogram(df, x=feature, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    if st.button("📉 Correlation Heatmap"):
        corr = df[numerical_features].corr()
        fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# Regression Section
st.sidebar.header("📊 Regression Model")
feature_selection = st.sidebar.multiselect("Select Features", options=numerical_features, default=numerical_features[:3])
target_variable = st.sidebar.selectbox("Select Target Variable", options=numerical_features)

if st.sidebar.button("Train Regression Model"):
    if not feature_selection or target_variable not in numerical_features:
        st.sidebar.error("Please select valid features and a target variable.")
    else:
        X = df[feature_selection]
        y = df[target_variable]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.subheader("Regression Model Summary")
        st.write(model.summary())
        
        st.subheader("Predictions vs Actuals")
        predictions = model.predict(X)
        result_df = pd.DataFrame({"Actual": y, "Predicted": predictions})
        st.write(result_df.head())
        
        fig = px.scatter(result_df, x="Actual", y="Predicted", title="Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

# Machine Learning Model - Logistic Regression (for Prediction)
st.sidebar.header("🧠 Machine Learning Model")
if st.sidebar.button("Train Logistic Model"):
    X = df.drop(columns=[target_variable])
    y = df[target_variable].astype(int) if df[target_variable].nunique() == 2 else None
    if y is None:
        st.sidebar.error("Target variable must be binary for logistic regression.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        joblib.dump(model, "logistic_model.pkl")
        st.sidebar.success("Logistic model trained successfully!")

st.sidebar.header("📥 Download Data")
st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False), "filtered_data.csv", "text/csv")
