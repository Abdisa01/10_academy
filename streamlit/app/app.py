import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_data, process_data, visual_data

def main():
    st.set_page_config(page_title="Solar Data Analysis", layout="wide")
    st.title("Solar Data Analysis App")
    st.write("Explore solar data and visualize key insights.")

    try:
        data = fetch_data("data/benin-malanville.csv")
        processed_data = process_data(data)

        # Interactive Visualization
        st.header("Interactive Visualization")
        selected_variable = st.selectbox("Select Variable", ["GHI", "DNI", "DHI"])
        st.line_chart(processed_data[selected_variable])

        # Summary Statistics
        st.header("Summary Statistics")
        st.dataframe(processed_data.describe())

        # Correlation Analysis
        st.header("Correlation Analysis")
        correlation_matrix = processed_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Histograms
        st.header("Histograms")
        st.bar_chart(processed_data[selected_variable].value_counts())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()