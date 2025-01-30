import streamlit as st
import pandas as pd
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title("Diabetes Prediction and Regression Dashboard")

# Upload CSV Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset into pandas dataframe
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(df.head())

    # Check if 'Outcome' column is present
    if 'Outcome' in df.columns:
        st.sidebar.success("'Outcome' column found for prediction.")
    else:
        st.sidebar.warning("No 'Outcome' column found for prediction.")

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

            # Make predictions on the test set
            y_pred = model.predict(X_test)
            st.write("Predictions on Test Data:")
            st.write(pd.DataFrame({"True Values": y_test, "Predictions": y_pred}))

            # Accuracy
            accuracy = (y_pred == y_test).mean()
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Confusion Matrix Plot
            cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        else:
            st.sidebar.warning("The dataset doesn't have a column named 'Outcome' for prediction.")

    # Regression Section - New Card with Full Model Output
    st.sidebar.header("📊 Regression Model")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()  # Dynamically get numeric columns
    if numeric_columns:
        feature_selection = st.sidebar.multiselect(
            "Select Features for Regression Model",
            options=numeric_columns,  # Automatically use available numeric columns
            default=numeric_columns[:min(4, len(numeric_columns))]  # Set default to first 4 or fewer columns
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
                
                # Correlation Heatmap
                correlation_matrix = df[feature_selection].corr()
                fig_corr, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig_corr)

                # Pairplot for selected features
                fig_pairplot = sns.pairplot(df[feature_selection])
                st.pyplot(fig_pairplot)

    else:
        st.sidebar.warning("The dataset doesn't have enough numeric columns for regression.")
    
else:
    st.sidebar.info("Please upload a CSV file to get started.")
