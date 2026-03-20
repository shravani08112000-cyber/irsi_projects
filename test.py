import streamlit as st
import pandas as pd

st.title("Data Explorer")
st.write("Upload CSV file and Explore your data")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    st.sidebar.header("Filters")
    column = st.sidebar.selectbox("Filter by column", df.columns)
    unique_vals = df[column].dropna().unique()
    selected = st.sidebar.multiselect("Select Values", unique_vals, default=unique_vals[:3])
    filter_df = df[df[column].isin(selected)]

    st.subheader('Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Filtered Rows",filter_df.shape[0])
    col3.metric("Columns", df.shape[1])

    st.subheader("Data")
    st.dataframe(filter_df)

    st.download_button(
        label="Download Filtered CSV",
        data = filter_df.to_csv(index=False),
        file_name='filter_data.csv',
        mime='text/csv'
    )

    st.subheader("Chart")
    if numeric_cols:
        chart_col = st.selectbox('Pick a numeric colmun to chart', numeric_cols)
        st.bar_chart(filter_df[chart_col])
    else:
        st.info("No columns found....")
else:
    st.info("Upload a CSV File to get started")