import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
col1, col2, col3 = st.columns(3)

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

# Regression Section - New Card
st.sidebar.header("📊 Regression Model")
feature_selection = st.sidebar.multiselect(
    "Select Features for Regression Model",
    options=df.columns[:-1],  # Exclude the Outcome column
    default=['Pregnancies', 'Glucose', 'BMI', 'Age']  # Default features
)

if st.sidebar.button("Train Regression Model"):
    # Features and target selection
    X = df[feature_selection]  # Use the selected features
    y = df['Glucose']  # Plasma Glucose concentration is the dependent variable
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate key metrics
    intercept = model.intercept_
    coefficients = model.coef_
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Display model details
    st.sidebar.success("Regression Model Trained Successfully!")
    
    # Display coefficients and intercept
    st.subheader("Regression Model Details")
    st.write(f"**Intercept:** {intercept:.2f}")
    st.write("**Coefficients for selected features:**")
    for feature, coef in zip(feature_selection, coefficients):
        st.write(f"{feature}: {coef:.2f}")
    
    st.write(f"**R-squared:** {r2:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    
    # Visualizing regression results
    st.subheader("Regression Predictions vs Actuals")
    regression_result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.write(regression_result_df.head())

    # Plotting actual vs predicted values
    fig = px.scatter(regression_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Plasma Glucose Concentration")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Plot residuals
    residuals = y_test - y_pred
    fig_residuals = px.scatter(x=y_test, y=residuals, title="Residuals Plot")
    st.plotly_chart(fig_residuals, use_container_width=True)

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

# Machine Learning Model - Logistic Regression (for Prediction)
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
