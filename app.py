import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app configuration
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# Sidebar for uploading dataset
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded dataset into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
        st.write(f"**Dataset Preview (First 5 Rows):**")
        st.write(df.head())
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {str(e)}")
        st.stop()

    # Dynamically update column names if necessary
    df.columns = df.columns.str.strip()  # Strip any extra spaces in column names

    # Ensure the dataset is not empty
    if df.empty:
        st.sidebar.warning("The uploaded dataset is empty.")
        st.stop()

else:
    st.sidebar.warning("Please upload a CSV file to start.")

# Apply the rest of the app logic only if dataset is available
if uploaded_file is not None and not df.empty:
    # Sidebar filters for dataset
    st.sidebar.header("🔍 Filter Data")
    
    # Check if the dataset has an 'Age' column, if not, skip the filter
    if 'Age' in df.columns:
        min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
        age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
        df_filtered = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]
    else:
        df_filtered = df

    # Create card layout
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Data Summary"):
            st.write(df_filtered.describe())

    with col2:
        if st.button("📊 Feature Distributions"):
            feature = st.selectbox("Select feature", df_filtered.select_dtypes(include=['number']).columns)
            if feature:
                fig = px.histogram(df_filtered, x=feature, color="Outcome", barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)

    with col3:
        if st.button("📉 Correlation Heatmap"):
            # Ensure you're working with numeric columns
            numeric_columns = df_filtered.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 1:
                corr = df_filtered[numeric_columns].corr()
                fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation.")

    # Regression Section - New Card with Full Model Output
    st.sidebar.header("📊 Regression Model")
    feature_selection = st.sidebar.multiselect(
        "Select Features for Regression Model",
        options=df.select_dtypes(include=['number']).columns.tolist(),  # Only select numeric features
        default=['Pregnancies', 'Glucose', 'BMI', 'Age']  # Default features if they exist
    )

    if feature_selection:
        if st.sidebar.button("Train Full Regression Model"):
            # Features and target selection
            X = df[feature_selection]  # Use the selected features
            if 'Outcome' in df.columns:
                y = df['Outcome'].apply(lambda x: 1 if x == 'positive' else 0)  # Assuming 'Outcome' is binary
            else:
                st.sidebar.warning("The dataset doesn't have a column named 'Outcome'.")
                st.stop()

            # Add constant for intercept
            X = sm.add_constant(X)

            # Train a linear regression model using statsmodels
            model = sm.OLS(y, X).fit()

            # Display regression model summary
            st.sidebar.success("Full Regression Model Trained Successfully!")
            st.subheader("Full Regression Model Summary")
            st.write(model.summary())

            # Display coefficients and intercept
            st.write(f"**Intercept:** {model.params[0]:.2f}")
            st.write("**Coefficients for selected features:**")
            for feature, coef in zip(feature_selection, model.params[1:]):
                st.write(f"{feature}: {coef:.2f}")

            st.write(f"**R-squared:** {model.rsquared:.2f}")
            st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y, model.predict(X)):.2f}")

            # Optional: Plot residuals
            residuals = y - model.predict(X)
            fig_residuals = px.scatter(x=y, y=residuals, title="Residuals Plot")
            st.plotly_chart(fig_residuals, use_container_width=True)

    # Machine Learning Model - Logistic Regression (for Prediction)
    st.sidebar.header("🧠 Diabetes Prediction")
    if st.sidebar.button("Train Model"):
        # Ensure there's a valid 'Outcome' column
        if 'Outcome' in df.columns:
            X = df.drop(columns=["Outcome"], errors='ignore')
            y = (df["Outcome"] == "positive").astype(int)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            joblib.dump(model, "diabetes_model.pkl")
            st.sidebar.success("Model trained successfully!")
        else:
            st.sidebar.warning("The dataset doesn't have a column named 'Outcome' for prediction.")

    st.sidebar.header("📡 Make Prediction")
    if st.sidebar.button("Load Model"):
        try:
            model = joblib.load("diabetes_model.pkl")
            inputs = [st.sidebar.number_input(f"{col}", value=float(df[col].mean())) for col in X.columns]
            prediction = model.predict([inputs])
            st.sidebar.write("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
        except FileNotFoundError:
            st.sidebar.warning("Model not found. Train the model first.")

    # Download option
    st.sidebar.header("📥 Download Data")
    st.sidebar.download_button("Download Filtered Data", df_filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
