import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================
# PROJECT HEADER
# ==========================

st.title("🛍️ Mall Customer Segmentation")

st.markdown(
    "### Machine Learning Based Customer Segmentation Dashboard"
)


# Display Image
try:
    image = Image.open("mall.jpg")

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        st.image(image, width=450)

except:
    pass

st.markdown("---")

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("dataset.csv")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Customer Segmentation"
    ]
)

# ==========================
# DATASET OVERVIEW PAGE
# ==========================

if page == "Dataset Overview":

    st.header("📖 About Dataset")

    st.write("""
    This dataset contains customer information collected from a shopping mall.

    Features:
    - Customer ID
    - Gender
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)

    The objective is to segment customers into different groups using the K-Means Clustering algorithm based on their spending behavior and income.
    """)

    st.header("📊 Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric("Recommended Clusters", 5)

    st.header("📋 Dataset Preview")

    st.dataframe(df.head(10))

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Gender Distribution
    st.header("👥 Gender Distribution")

    gender_count = df["Gender"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        gender_count,
        labels=gender_count.index,
        autopct="%1.1f%%"
    )

    ax.set_title("Gender Distribution")

    st.pyplot(fig)

    # Age Distribution
    st.header("🎂 Age Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["Age"],
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Customer Ages")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # Income vs Spending
    st.header("💰 Annual Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        s=80,
        ax=ax
    )

    ax.set_title("Customer Spending Behavior")

    st.pyplot(fig)

# ==========================
# CUSTOMER SEGMENTATION PAGE
# ==========================

if page == "Customer Segmentation":

    st.header("🎯 Customer Segmentation Results")

    clusters = st.sidebar.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=5
    )

    # Feature Selection
    X = df[
        [
            'Annual Income (k$)',
            'Spending Score (1-100)'
        ]
    ]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Clustered Data
    st.subheader("📋 Clustered Customer Data")

    st.dataframe(df)

    # Cluster Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set1",
        s=100,
        ax=ax
    )

    ax.set_title(
        f"Customer Segmentation using K-Means (K={clusters})"
    )

    st.pyplot(fig)

    # Customers per Cluster
    st.subheader("👥 Customers per Cluster")

    cluster_count = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_count)

    # Cluster Summary
    st.subheader("📊 Cluster Summary")

    summary = df.groupby(
        "Cluster"
    ).mean(
        numeric_only=True
    )

    st.dataframe(summary)

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        Developed by <b>Dhanu Sree</b> |
        SkillCraft Technology ML Internship <br>
    </div>
    """,
    unsafe_allow_html=True
)
