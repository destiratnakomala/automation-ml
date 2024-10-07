import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

# Main function
def main():
    st.set_page_config(page_title="Data Automation-Machine Learning")
    st.title("Machine Learning")

    uploaded_file = None
    data = None

    # Step 1: Upload Data
    with st.expander("1: Add Your Data Source"):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    with st.expander("2: DataSet Preview"):
        if uploaded_file is not None:

            data = pd.read_csv(uploaded_file)
            # Step 2: Data Overview
            view1, view2,view3, view4 = st.columns(4)
            with view1: 
                st.write("Data Overview")
                st.dataframe(data.head())
            with view2:
                st.write(" Data Description")
                st.write(data.describe())
            with view3:
                st.write(" Missing Values")
                st.write(data.isnull().sum())
            with view4:
                st.write(" Data Types")
                st.write(data.dtypes)


    with st.expander("3: Data Cleaning"):           
            # Step 3: Data Cleaning
            clean1, clean2, clean3 = st.columns(3)
            with clean1: 
                st.write(" Data Summary Before Cleaning")
                st.write(data.describe())
            with clean2:
                st.write("Missing Values Before Cleaning:")
                st.write(data.isnull().sum())
            with clean3:
                # Visualize missing values
                if st.checkbox("Show Missing Values Heatmap"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax)
                    plt.title("Missing Values Heatmap")
                    st.pyplot(fig)

            clean4, clean5= st.columns(2)
            with clean4:
                # Remove duplicates
                if st.checkbox("Remove Duplicate Rows"):
                    initial_shape = data.shape
                    data = data.drop_duplicates()
                    st.success(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")


            with clean5:
            # Handle missing values
                missing_strategy = st.selectbox(
                    "Choose a strategy for handling missing values",
                    options=["Drop Missing Values", "Fill with Mean", "Fill with Median", "Fill with Mode", "Do Nothing"]
                )

                if st.button("Apply Missing Value Strategy"):
                    if missing_strategy == "Drop Missing Values":
                        data.dropna(inplace=True)
                        st.success("Dropped rows with missing values.")
                    elif missing_strategy == "Fill with Mean":
                        data.fillna(data.mean(), inplace=True)
                        st.success("Filled missing values with the mean.")
                    elif missing_strategy == "Fill with Median":
                        data.fillna(data.median(), inplace=True)
                        st.success("Filled missing values with the median.")
                    elif missing_strategy == "Fill with Mode":
                        for column in data.select_dtypes(include=['object']).columns:
                            data[column].fillna(data[column].mode()[0], inplace=True)
                        st.success("Filled missing values with the mode for categorical columns.")
                    elif missing_strategy == "Do Nothing":
                        st.info("No changes made to missing values.")
            clean7, clean8= st.columns(2)
            with clean7:
                # Display basic info after cleaning
                st.write(" Data Summary After Cleaning")
                st.write(data.describe())
            with clean8:
                st.write("Missing Values After Cleaning:")
                st.write(data.isnull().sum())
    
    with st.expander('4: EDA'):
            
            # Step 4: Exploratory Data Analysis (EDA)
            st.write("Correlation Matrix")

            # Calculate the correlation matrix
            correlation_matrix = data.corr()

            # Create a heatmap using Plotly
            fig = ff.create_annotated_heatmap(
                z=correlation_matrix.values,
                x=list(correlation_matrix.columns),
                y=list(correlation_matrix.index),
            )

            # Update layout for better readability
            fig.update_layout(
                title="Correlation Matrix",
                xaxis_title="Features",
                yaxis_title="Features",
                width=700,  # Adjust width as needed
                height=500,  # Adjust height as needed
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig)
            eda1, eda2= st.columns(2)
            with eda1:
                # Plotting distributions for numerical features
                if st.checkbox("Show Distribution Plots for Numeric Features"):
                    for column in data.select_dtypes(include=[int, float]).columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(data[column], bins=30, kde=True, ax=ax)
                        plt.title(f'Distribution of {column}')
                        st.pyplot(fig)
            with eda2:
                # Boxplots for outlier detection
                if st.checkbox("Show Boxplots for Numeric Features"):
                    for column in data.select_dtypes(include=[int, float]).columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.boxplot(x=data[column], ax=ax)
                        plt.title(f'Boxplot of {column}')
                        st.pyplot(fig)

    with st.expander("5: Feature Engineering"):           
            target_column = st.selectbox("Select the target variable", options=data.columns)
            feature_columns = st.multiselect("Select features", options=data.columns.drop(target_column))
    with st.expander("6: Modelling "):
            # Initialize session state for storing results
            if 'model_plot' not in st.session_state:
                st.session_state.model_plot = None
            if 'model_metrics' not in st.session_state:
                st.session_state.model_metrics = None

            # Model training
            model_option = st.selectbox("Select Regression Model", options=["Linear Regression", "Random Forest Regression", "Lasso Regression"])

            if st.button("Train Model (Without Hyperparameter Tuning)"):
                if feature_columns:
                    X = data[feature_columns]
                    y = data[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Initialize the selected model
                    if model_option == "Linear Regression":
                        model = LinearRegression()
                    elif model_option == "Random Forest Regression":
                        model = RandomForestRegressor(random_state=42)
                    elif model_option == "Lasso Regression":
                        model = Lasso()

                    # Train model
                    model.fit(X_train, y_train)

                    # Save the model
                    model_name = st.text_input('Enter model name', 'my_model')
                    model_file_path = f'{model_name}.pkl'
                    joblib.dump(model, model_file_path)
                    st.success("Model saved successfully!")

                    # Add a download button for the model
                    with open(model_file_path, "rb") as f:
                        st.download_button(
                            label="Download Model",
                            data=f,
                            file_name=model_file_path,
                            mime="application/octet-stream"
                        )

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Step 7: Visualization of Predictions (Line Plot)
                    st.session_state.model_plot = (y_test.reset_index(drop=True), y_pred)
                    st.session_state.model_metrics = (mse, r2)

                    # Show results
                    st.success(f"Mean Squared Error: {mse:.2f}")
                    st.success(f"R^2 Score: {r2:.2f}")




            # Display model plot if available
            if st.session_state.model_plot is not None:
                y_test, y_pred = st.session_state.model_plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(y_test, label="True Values", color="blue", linestyle="--")
                ax.plot(y_pred, label="Predicted Values", color="orange")
                ax.set_title(f'{model_option}: True Values vs Predictions')
                ax.set_xlabel('Index')
                ax.set_ylabel('Values')
                ax.legend()
                st.pyplot(fig)

                # Display metrics if available
                if st.session_state.model_metrics is not None:
                    mse, r2 = st.session_state.model_metrics
                    st.success(f"Mean Squared Error: {mse:.2f}")
                    st.success(f"R^2 Score: {r2:.2f}")


    with st.expander("7: HyperParameter"):
            # Step 8: Hyperparameter Tuning
            st.write("Hyperparameter Tuning")
            if feature_columns:
                hyperparam_model_option = st.selectbox("Select Model for Hyperparameter Tuning", options=["Linear Regression", "Random Forest Regression", "Lasso Regression"])

                if hyperparam_model_option == "Linear Regression":
                    param_grid = {'fit_intercept': [True, False]}
                elif hyperparam_model_option == "Random Forest Regression":
                    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
                elif hyperparam_model_option == "Lasso Regression":
                    param_grid = {'alpha': [0.01, 0.1, 1, 10], 'max_iter': [1000, 5000, 10000]}

                if st.button("Train Model with Hyperparameter Tuning"):
                    # Prepare data for training
                    X = data[feature_columns]
                    y = data[target_column]

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Initialize and perform hyperparameter tuning
                    if hyperparam_model_option == "Linear Regression":
                        model = LinearRegression()
                        grid_search = GridSearchCV(model, param_grid, cv=5)
                    elif hyperparam_model_option == "Random Forest Regression":
                        model = RandomForestRegressor(random_state=42)
                        grid_search = GridSearchCV(model, param_grid, cv=5)
                    elif hyperparam_model_option == "Lasso Regression":
                        model = Lasso()
                        grid_search = GridSearchCV(model, param_grid, cv=5)

                    # Train the model
                    grid_search.fit(X_train, y_train)

                    # Make predictions
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Step 9: Visualization of Predictions (Line Plot)
                    st.session_state.model_plot = (y_test.reset_index(drop=True), y_pred)
                    st.session_state.model_metrics = (mse, r2)

                    # Show results
                    st.success(f"Best Parameters: {grid_search.best_params_}")
                    st.success(f"Mean Squared Error: {mse:.2f}")
                    st.success(f"R^2 Score: {r2:.2f}")

                # Display hyperparameter tuned model plot if available
                if st.session_state.model_plot is not None:
                    y_test, y_pred = st.session_state.model_plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(y_test, label="True Values", color="blue", linestyle="--")
                    ax.plot(y_pred, label="Predicted Values", color="orange")
                    ax.set_title(f'{hyperparam_model_option}: True Values vs Predictions (Tuned)')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Values')
                    ax.legend()
                    st.pyplot(fig)

                    # Display metrics if available
                    if st.session_state.model_metrics is not None:
                        mse, r2 = st.session_state.model_metrics
                        st.success(f"Mean Squared Error: {mse:.2f}")
                        st.success(f"R^2 Score: {r2:.2f}")



# Run the app
if __name__ == "__main__":
    main()
