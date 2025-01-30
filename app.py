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

# Streamlit app configuration
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# Sidebar for uploading dataset
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded dataset into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
    st.write(f"**Dataset Preview (First 5 Rows):**")
    st.write(df.head())

    # Dynamically update column names if necessary
    if "Outcome" not in df.columns:
        st.sidebar.warning("The uploaded dataset doesn't have a column named 'Outcome'. Please ensure your data includes this column.")

    # Rename columns for consistency if necessary
    df.columns = df.columns.str.strip()  # Strip any extra spaces in column names
else:
    st.sidebar.warning("Please upload a CSV file to start.")

# Apply the rest of the app logic only if dataset is available
if uploaded_file is not None:
    # Sidebar filters for dataset
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
            feature = st.selectbox("Select feature", df_filtered.columns[:-1])  # Exclude the Outcome column
            fig = px.histogram(df_filtered, x=feature, color="Outcome", barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if st.button("📉 Correlation Heatmap"):
            # Ensure you're working with numeric columns
            numeric_columns = df_filtered.select_dtypes(include=['number']).columns
            corr = df_filtered[numeric_columns].corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # Regression Section - New Card with Full Model Output
    st.sidebar.header("📊 Regression Model")
    feature_selection = st.sidebar.multiselect(
        "Select Features for Regression Model",
        options=df.columns[:-1],  # Exclude the Outcome column
        default=['Pregnancies', 'Glucose', 'BMI', 'Age']  # Default features
    )

    if st.sidebar.button("Train Full Regression Model"):
        # Features and target selection
        X = df[feature_selection]  # Use the selected features
        y = df['Glucose']  # Plasma Glucose concentration is the dependent variable

        # Add constant for intercept
        X = sm.add_constant(X)

        # Train a linear regression model using statsmodels
        model = sm.OLS(y, X).fit()

        # Get p-values and confidence intervals
        p_values = model.pvalues
        conf_int = model.conf_int()

        # Display full model details
        st.sidebar.success("Full Regression Model Trained Successfully!")

        # Display regression model summary
        st.subheader("Full Regression Model Summary")
        st.write(model.summary())

        # Display coefficients and intercept
        st.write(f"**Intercept:** {model.params[0]:.2f}")
        st.write("**Coefficients for selected features:**")
        for feature, coef in zip(feature_selection, model.params[1:]):
            st.write(f"{feature}: {coef:.2f}")

        st.write(f"**R-squared:** {model.rsquared:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y, model.predict(X)):.2f}")

        # Display p-values and confidence intervals
        st.write("**P-values for selected features:**")
        for feature, p_val in zip(feature_selection, p_values[1:]):
            st.write(f"{feature}: {p_val:.4f}")

        # Display Confidence Intervals
        st.write("**Confidence Intervals (95%) for selected features:**")
        for feature, conf in zip(feature_selection, conf_int.values[1:]):
            st.write(f"{feature}: ({conf[0]:.2f}, {conf[1]:.2f})")

        # Visualizing regression results
        st.subheader("Regression Predictions vs Actuals")
        regression_result_df = pd.DataFrame({
            "Actual": y,
            "Predicted": model.predict(X)
        })
        st.write(regression_result_df.head())

        # Plotting actual vs predicted values
        fig = px.scatter(regression_result_df, x="Actual", y="Predicted", title="Actual vs Predicted Plasma Glucose Concentration")
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Plot residuals
        residuals = y - model.predict(X)
        fig_residuals = px.scatter(x=y, y=residuals, title="Residuals Plot")
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

