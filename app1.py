import base64
from contextlib import nullcontext
from queue import Full
from ssl import Options
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
from PIL import Image
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.set_page_config(page_title="Data Dumper",
                   page_icon=":bar_chart:", layout="wide")

st.title(":chart_with_upwards_trend: Insight analysis ")
st.image("https://i.pinimg.com/736x/03/66/39/0366391ef604d1f4514389023f2dc4f8.jpg",
         width=1000)

st.sidebar.header(" About Us ")
st.sidebar.write(
    "Without big data analytics,Companies are blind and deaf, wandering out onto the web like a deer on the freeway -Geoffrey Moore")
st.sidebar.write("Writing up the results of a data analysis is not a skill that anyone is born with. It requires practice and, at least in the beginning, a bit of guidance.  Thus we provide easy analysis for the users without the knowledge of coding so that they can infer knowledge from existing data.")
st.sidebar.subheader(" Contact")
st.sidebar.write(":telephone_receiver: 9345815421")
st.sidebar.write(":e-mail: susanthnarayananr@gmail.com")
st.sidebar.write(":iphone: www.linkedin.com/in/susanth-narayanan-r")
st.sidebar.write(":round_pushpin: Coimbatore,India")


st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
}
</style>
    """, unsafe_allow_html=True)


st.write("Data Is the future and the future is now!Every mouse click,keyboard button press,swipe or tap is used to shape business decisions. Everything is about data these days. Data is information and Information is power")
st.write("Services offered are as follows")
st.write("1) Checking and filling null values (cleaning data)")
st.write("2) Standardizing data")
st.write("3) Data Description")
st.write("4) Data Visualization")
st.write("5) Statistical Analysis")


uploaded_file = st.file_uploader("Upload a file here")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("The given data is... :")
    st.write(df)

    # cleaning Data

    st.write("The checking null values from the given data is...:")
    st.write(df.isnull().sum())
    n = df.fillna(df.mean())
    st.write("After filling null values ", df.fillna(df.mean()))
    st.write("Click to download file")

    def get_table_download_link_csv(n):
        csv = n.to_csv().encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="fillednull.csv" target="_blank">Download csv file with filled null values</a>'
        return href
    st.markdown(get_table_download_link_csv(n), unsafe_allow_html=True)

    # Standardizing Data

    st.write("Standardizing data")

    normalized_df = (df-df.mean())/df.std()
    st.write("Data standardization is the process of converting data to a common format to enable processing and analyzing it. Most organizations utilize data from a number of sources; this can include data warehouses, lakes, cloud storage, and databases.", normalized_df)

    def get_table_download_link_csv(normalized_df):
        csv = normalized_df.to_csv().encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="standard.csv" target="_blank">Download Standardized dataset</a>'
        return href
    st.markdown(get_table_download_link_csv(
        normalized_df), unsafe_allow_html=True)

    # Encoding Data

    data = df
    label_encoders = {}
    categorical_columns = df.columns
    for columns in categorical_columns:
        label_encoders[columns] = LabelEncoder()
        df[columns] = label_encoders[columns].fit_transform(df[columns])

    st.write("LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels. Fit label encoder. Fit label encoder and return encoded labels.")
    st.write("Encoding data", data)

    def get_table_download_link_csv(data):
        csv = data.to_csv().encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="labled.csv" target="_blank">Download Encoded dataset</a>'
        return href
    st.markdown(get_table_download_link_csv(data), unsafe_allow_html=True)

    # st.write("COLUMNS Available are :", numerical_cols)
    # st.write("Numerical columns are", df.select_dtypes(include=np.number))
    st.write("If the DataSet contains numerical data, the description contains these information for each column:  count - The number of not-empty values. mean - The average (mean) value. std - The standard deviation. min - the minimum value. 25% - The 25% percentile. 50% - The 50% percentile. 75% - The 75% percentile. max - the maximum value.")

    st.write("Description values in each columns:", df.describe())

    st.write(
        "The Maximum minimum and average values in each columns are :")

    tab1, tab2, tab3 = st.tabs(["Max", "Min", "Average"])

    with tab1:
        st.header("Maximum")
        st.write("Maximum values in each columns:", df.max(numeric_only=True))

    with tab2:
        st.header("Minimum")
        st.write("Minimum values in each columns:", df.min(numeric_only=True))

    with tab3:
        st.header("Average")
        st.write("Average values in each columns:", df.mean(numeric_only=True))

    fig, ax = plt.subplots(figsize=(5, 5))
    sn.heatmap(df.corr(), ax=ax)
    st.write("A heatmap ( or heat map) is a graphical representation of data where values are depicted by color./n  They are essential in detecting what does or doesn't work on a website or product page. By experimenting with how certain buttons and elements are positioned on your website, heatmaps allow you to evaluate your product’s performance and increase user engagement and retention as you prioritize the jobs to be done that boost customer value. Heatmaps make it easy to visualize complex data and understand it at a glance:"

             )
    st.write("Heatmap", fig)

    st.write("Statistical Tests are as follows:")

    tab1, tab2 = st.tabs(["Sharpio", "Gaussian"])

    with tab1:
        st.header("Sharpio")
        st.write("Sharpiro Test:")
        st.write("The Shapiro–Wilk test can be used to decide whether or not a sample fits a normal distribution, and it is commonly used for small samples.")
        name = st.text_input(
            'Enter a column name to continue with statistical tests: ',)
        nor_test = df[name].tolist()
        stat, p = shapiro(nor_test)
        st.write('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            st.write('Probably Gaussian')
        else:
            st.write('Probably not Gaussian')

    with tab2:

        st.header("Pearsons Correlation Test")

        st.write("Pearsons Correlation Test")

        st.write("Pearson’s correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables.  It is known as the best method of measuring the association between variables of interest because it is based on the method of covariance.  It gives information about the magnitude of the association, or correlation, as well as the direction of the relationship.")
        dep1 = st.text_input(
            'Enter a depandant column name : ',)
        dep2 = st.text_input(
            'Enter a independant column name : ',)
        per_test1 = df[dep1].tolist()
        per_test2 = df[dep2].tolist()
        stat, p = pearsonr(per_test1, per_test2)
        st.write('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably independent')
        else:
            print('Probably dependent')
