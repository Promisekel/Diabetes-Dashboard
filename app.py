import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("diabetesData.csv")

# Rename columns for consistency
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

# Streamlit app
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
df_filtered = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]

# Create card layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Data Summary"):
        st.write(df_filtered.describe())

with col2:
    if st.button("📊 Feature Distributions"):
        feature = st.selectbox("Select feature", df_filtered.columns[:-1])
        fig = px.histogram(df_filtered, x=feature, color="Outcome", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    if st.button("📉 Correlation Heatmap"):
        corr = df_filtered.corr()
        fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
        st.plotly_chart(fig, use_container_width=True)

with col4:
    if st.button("📐 Regression Model"):
        # Prepare data for regression
        X = df_filtered.drop(columns=["Outcome"])
        X = sm.add_constant(X)  # Adds an intercept to the model
        y = df_filtered["Outcome"]

        # Fit Logistic Regression model using Statsmodels
        model = sm.Logit(y, X)
        result = model.fit()

        # Display full regression model results
        st.write("**Full Logistic Regression Model Results:**")
        st.write(result.summary())

        # Extracting p-values and confidence intervals
        st.write("**Confidence Intervals (95%) for selected features:**")
        conf_int = result.conf_int(alpha=0.05)  # 95% confidence intervals

        # Loop through the features and display confidence intervals
        for feature, conf in zip(X.columns, conf_int.values):
            st.write(f"{feature}: ({conf[0]:.2f}, {conf[1]:.2f})")

        st.write("**P-values for selected features:**")
        pvalues = result.pvalues
        for feature, p_value in pvalues.items():
            st.write(f"{feature}: {p_value:.4f}")

# Additional visualizations
st.sidebar.header("📌 More Insights")
option = st.sidebar.radio("Select a visualization:", ["Boxplot", "Diabetes Count", "Age Distribution", "Pair Plot"])

if option == "Boxplot":
    feature = st.sidebar.selectbox("Choose feature", df_filtered.columns[:-1])
    fig = px.box(df_filtered, x="Outcome", y=feature, color="Outcome")
    st.plotly_chart(fig, use_container_width=True)

elif option == "Diabetes Count":
    fig = px.bar(df_filtered["Outcome"].value_counts(), x=df_filtered["Outcome"].unique(), y=df_filtered["Outcome"].value_counts(), color=df_filtered["Outcome"].unique())
    st.plotly_chart(fig, use_container_width=True)

elif option == "Age Distribution":
    fig = px.histogram(df_filtered, x="Age", color="Outcome", nbins=20, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

elif option == "Pair Plot":
    fig = px.scatter_matrix(df_filtered, dimensions=['Glucose', 'BloodPressure', 'BMI', 'Age'], color='Outcome')
    st.plotly_chart(fig, use_container_width=True)

# Machine Learning Model - Logistic Regression
st.sidebar.header("🧠 Diabetes Prediction")
if st.sidebar.button("Train Model"):
    X = df.drop(columns=["Outcome"])
    y = (df["Outcome"] == "positive").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    joblib.dump(model, "diabetes_model.pkl")
    st.sidebar.success("Model trained successfully!")

st.sidebar.header("📡 Make Prediction")
if st.sidebar.button("Load Model"):
    model = joblib.load("diabetes_model.pkl")
    inputs = [st.sidebar.number_input(f"{col}", value=float(df[col].mean())) for col in X.columns]
    prediction = model.predict([inputs])
    st.sidebar.write("Prediction:", "Positive" if prediction[0] == 1 else "Negative")

# Download option
st.sidebar.header("📥 Download Data")
st.sidebar.download_button("Download Filtered Data", df_filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
