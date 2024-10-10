import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


customer_data = pd.read_csv('customer_churn.csv')

def predict_churn(features):
    url="https://2l01x6r0cc.execute-api.us-east-1.amazonaws.com/dev"
    response = requests.post(url, json=features)
    return response.json()["prediction"]

with st.sidebar:
    page = st.selectbox("Page", ["Predictor", "EDA"])

if page == "Predictor":
    st.title("Customer Churn Prediction")

    age = st.number_input("Age", 22, 65)
    total_purchase = st.number_input("Total Purchase", 100.0, 18026.01)
    account_manager = st.selectbox("Has Account Manager", ["Yes", "No"])
    years = st.slider("Years", 1.0, 9.15)
    num_sites = st.slider("No. of Sites", 3, 14)

    if st.button("Predict Churn"):
        features = {
            "age" : age,
            "total_purchase" :total_purchase,
            "account_manager" :1 if account_manager == "Yes" else 0,
            "years" :years,
            "num_sites" : num_sites,
        }

        prediction = predict_churn(features)

        if prediction == 1:
            st.error("This customer is likely to churn!")
        else:
            st.success("This customer is unlikely to churn.")


if page == "EDA":
    st.header("Introduction")
    st.write("""
    The marketing agency is facing high customer churn. Our goal is to create a machine learning model that can predict 
    whether a customer will churn based on historical data. This will help the company assign account managers to at-risk customers.
    The dataset contains customer demographics and behavior data such as the number of ads purchased, account manager assignment, and number of years as a customer.
    """)

    st.header("Exploratory Data Analysis (EDA)")


    st.subheader("Dataset Overview")
    st.write("Here are the first few rows of the dataset:")
    st.dataframe(customer_data.head())


    st.subheader("Churn Distribution")
    churn_count = customer_data['Churn'].value_counts()
    st.bar_chart(churn_count)
    st.write("The dataset contains more customers who have not churned, creating an imbalance between the churned and non-churned classes.")


    st.subheader("Correlation Heatmap")
    num_data = customer_data.drop(columns=['Names','Onboard_date','Location','Company'])
    correlation_matrix = num_data.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.write("""
    From the heatmap, we observe a moderate positive correlation between the number of sites (Num_Sites) and churn.
    Other features like `Total_Purchase` and `Age` show weaker relationships with churn.
    """)

    st.subheader("Years as Customer vs Number of Sites by Churn")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Years', y='Num_Sites', hue='Churn', data=customer_data, palette='viridis', ax=ax)
    ax.set_title('Years as a Customer vs Number of Sites by Churn')
    ax.set_xlabel('Years as a Customer')
    ax.set_ylabel('Number of Sites')
    st.pyplot(fig)
    st.write("Customers with more websites using the service are more likely to churn, especially those with a longer tenure.")


    st.subheader("Total Purchase by Churn")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Churn', y='Total_Purchase', data=customer_data, palette='viridis', ax=ax)
    ax.set_title('Total Purchase by Churn')
    ax.set_xlabel('Churn (0 = No, 1 = Yes)')
    ax.set_ylabel('Total Purchase')
    st.pyplot(fig)
    st.write("There is no significant difference in total purchases between churned and non-churned customers.")



    st.header("XGBoost Model Results")
    st.subheader("Model Overview")
    st.write("""
    The XGBoost model was used to predict customer churn based on the features from the dataset. XGBoost was chosen due to its strong 
    performance on structured datasets and its ability to handle non-linear relationships.
    """)

    st.subheader("Model Performance")
    st.write(f"**Training Accuracy**: {(1-0.093056) * 100:.2f}%")
    st.write(f"**Testing Accuracy**: {(1-0.122222) * 100:.2f}%")
    st.write("""
    The model demonstrates a solid performance based on the accuracy metrics. With a training accuracy of 
approximately 90.69%, and a testing accuracy of 87.78%, the model is able to generalize well from the training data 
to unseen test data.""")


    # Conclusion
    st.header("Conclusion")
    st.write("""
    The analysis and model results show that customer churn can be effectively predicted using features like the number of websites and years as a customer.
    By implementing this model, the company can proactively assign account managers to high-risk customers, potentially reducing churn and improving customer retention.
    """)





