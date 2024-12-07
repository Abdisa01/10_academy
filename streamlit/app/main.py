# app/main.py
# app/main.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import fetch_data, process_data

def main():
    st.title("Solar Radiance Data Dashboard")

    # File upload or static CSV file usage
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = fetch_data(uploaded_file)
    else:
        # Load static file if no upload
        df = fetch_data("data/benin-malanville.csv")  # Use the correct file name here

    processed_data = process_data(df)

    # Data Display
    if st.checkbox("Show Data Table"):
        st.write(processed_data)

    # Visualizations
    st.header("Visualizations")

    # Plot GHI over time
    st.subheader("Global Horizontal Irradiance over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(processed_data.index, processed_data['GHI'], label='GHI', color='orange')
    plt.xlabel('Timestamp')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.title('GHI over Time')
    plt.legend()
    st.pyplot(plt)

    # Correlation Heatmap
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    correlation_matrix = processed_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # Additional visualizations can go here...

if __name__ == "__main__":
    main()