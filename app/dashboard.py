import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Load data
df = pd.read_csv("data/superstore.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# Feature engineering
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month_Name'] = df['Order Date'].dt.month_name()
df['Quarter'] = df['Order Date'].dt.quarter
festival_months = [10, 11, 12] 
df['Festival_Month'] = df['Month'].apply(lambda x: 1 if x in festival_months else 0)
df['Order_Value'] = df['Sales']

# Load model
model = joblib.load("models/sales_model.pkl")

# Sidebar
st.sidebar.title("Dashboard Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Revenue Trends", "Product Analysis", "Customer Behaviour", "Sales Prediction"]
)

st.title("Retail Sales Data Analysis and Forecasting Dashboard")
st.markdown("Interactive dashboard for sales trends, products, customer behaviour, and prediction.")

#overview section
if section == "Overview":
    st.header("Overview")

    total_sales = df['Sales'].sum()
    total_orders = df['Order ID'].nunique()
    total_customers = df['Customer ID'].nunique()
    avg_order_value = df['Sales'].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)
    col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=yearly_sales, x='Year', y='Sales', ax=ax)
    ax.set_title("Yearly Sales")
    st.pyplot(fig)
#revenue section
elif section == "Revenue Trends":
    st.header("Revenue Trends")

    monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly_sales, x='Month', y='Sales', hue='Year', marker='o', ax=ax)
    ax.set_title("Monthly Sales Trend")
    st.pyplot(fig)

    quarterly_sales = df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=quarterly_sales, x='Quarter', y='Sales', hue='Year', ax=ax)
    ax.set_title("Quarterly Sales Trend")
    st.pyplot(fig)

    festival_sales = df.groupby('Festival_Month')['Sales'].sum().reset_index()
    festival_sales['Festival_Label'] = festival_sales['Festival_Month'].map({0: 'Non-Festival', 1: 'Festival'})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=festival_sales, x='Festival_Label', y='Sales', ax=ax)
    ax.set_title("Festival vs Non-Festival Sales")
    st.pyplot(fig)

#product analysis
elif section == "Product Analysis":
    st.header("Product Analysis")

    top_products_sales = (
        df.groupby('Product Name')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_products_sales, x='Sales', y='Product Name', ax=ax)
    ax.set_title("Top 10 Products by Sales")
    st.pyplot(fig)

    top_products_orders = (
        df.groupby('Product Name')['Order ID']
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_products_orders, x='Order ID', y='Product Name', ax=ax)
    ax.set_title("Top 10 Products by Number of Orders")
    st.pyplot(fig)

    category_sales = df.groupby('Category')['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=category_sales, x='Category', y='Sales', ax=ax)
    ax.set_title("Sales by Category")
    st.pyplot(fig)

#customer behaviour section
elif section == "Customer Behaviour":
    st.header("Customer Behaviour")

    customer_sales = (
        df.groupby('Customer ID')['Sales']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_customers = customer_sales.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_customers, x='Sales', y='Customer ID', ax=ax)
    ax.set_title("Top 10 Customers by Total Spending")
    st.pyplot(fig)

    customer_orders = (
        df.groupby('Customer ID')['Order ID']
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_orders = customer_orders.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_orders, x='Order ID', y='Customer ID', ax=ax)
    ax.set_title("Top Customers by Number of Orders")
    st.pyplot(fig)

    customer_order_count = df.groupby('Customer ID')['Order ID'].nunique()
    repeat_customers = (customer_order_count > 1).sum()
    one_time_customers = (customer_order_count == 1).sum()

    pie_data = pd.DataFrame({
        'Customer Type': ['Repeat', 'One-Time'],
        'Count': [repeat_customers, one_time_customers]
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(pie_data['Count'], labels=pie_data['Customer Type'], autopct='%1.1f%%')
    ax.set_title("Repeat vs One-Time Customers")
    st.pyplot(fig)

#sales predction section
elif section == "Sales Prediction":
    st.header("Sales Prediction")

    st.subheader("Enter prediction inputs")

    prev_month_sales = st.number_input("Previous Month Sales", min_value=0.0, value=50000.0)
    prev_month_orders = st.number_input("Previous Month Orders", min_value=0, value=100)
    month = st.selectbox("Month", list(range(1, 13)))
    festival_month = st.selectbox("Festival Month", [0, 1])

    input_data = pd.DataFrame({
        'Prev_Month_Sales': [prev_month_sales],
        'Prev_Month_Orders': [prev_month_orders],
        'Month': [month],
        'Festival_Month': [festival_month]
    })

    if st.button("Predict Sales"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Sales: ${prediction:,.2f}")