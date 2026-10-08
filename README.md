# MCA-labsheet-7-
# Project 1: Customer Segmentation using K-Means Clustering.

# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
df.head()

print("Shape of dataset:", df.shape)
df.info()

df.describe()

df.isnull().sum()

# select features 
features = df[["Age","Annual Income (k$)","Spending Score (1-100)"
]]
features.head()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]

# Elbow Method - Determining the Optimal Number of Clusters
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Applying K-Means Clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
df["Cluster"] = kmeans.fit_predict(scaled_features)
df.head()

# cluster Evaluation 
silhouette = silhouette_score(
    scaled_features,
    df["Cluster"]
)
print("Silhouette Score:", round(silhouette, 3))

# visualization 
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# cluster Analysis 
cluster_summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()
cluster_summary

# Number of Customers in Each Cluster
df["Cluster"].value_counts().sort_index()

# Business Insights
- Customers are divided into five different groups.
- The clusters represent customers with different age, income and spending patterns.
- High-income and high-spending customers are valuable target customers.
- Low-income and low-spending customers may require different marketing strategies.
- Customer segmentation can help businesses design targeted marketing campaigns.

# Project 2: News Article Clustering System 
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_json(
    "News_Category_Dataset_v3.json",
    lines=True
)
df.head()

print("Shape of dataset:", df.shape)
df.info()

news_df = df[["headline","short_description","category"
]].copy()
news_df.head()

news_df.isnull().sum()

news_df["headline"] = news_df["headline"].fillna("")
news_df["short_description"] = news_df["short_description"].fillna("")
news_df["text"] = (
    news_df["headline"] + " " +
    news_df["short_description"]
)
news_df.head()

news_sample = news_df.sample(
    n=5000,
    random_state=42
).reset_index(drop=True)
print("Sample size:", news_sample.shape)
news_sample.head()

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
tfidf_matrix = vectorizer.fit_transform(
    news_sample["text"]
)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

scores = []
for k in range(2, 8):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )    
    labels = kmeans.fit_predict(tfidf_matrix)    
    score = silhouette_score(
        tfidf_matrix,
        labels
    )    
    scores.append(score)
for k, score in zip(range(2, 8), scores):
    print(
        "Clusters:", k,
        "Silhouette Score:", round(score, 3)
    )

best_k = range(2, 8)[np.argmax(scores)]
print("Best Number of Clusters:", best_k)

# K-Means Clustering
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
news_sample["KMeans_Cluster"] = kmeans.fit_predict(
    tfidf_matrix
)
news_sample[
    ["headline", "category", "KMeans_Cluster"]
].head(10)

kmeans_score = silhouette_score(
    tfidf_matrix,
    news_sample["KMeans_Cluster"]
)
print(
    "K-Means Silhouette Score:",
    round(kmeans_score, 3)
)

# Visualizing K-Means Clusters
svd = TruncatedSVD(
    n_components=2,
    random_state=42
)
reduced_data = svd.fit_transform(tfidf_matrix)
plt.figure(figsize=(9, 6))
plt.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    c=news_sample["KMeans_Cluster"],
    s=20
)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("News Article Clusters using K-Means")
plt.show()

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(
    n_clusters=best_k
)
hierarchical_labels = hierarchical.fit_predict(
    tfidf_matrix.toarray()
)
news_sample["Hierarchical_Cluster"] = hierarchical_labels
news_sample[
    ["headline", "category", "Hierarchical_Cluster"]
].head(10)

hierarchical_score = silhouette_score(
    tfidf_matrix,
    hierarchical_labels
)
print(
    "Hierarchical Clustering Silhouette Score:",
    round(hierarchical_score, 3)
)

comparison = pd.DataFrame({
    "Technique": [
        "K-Means",
        "Hierarchical Clustering"
    ],
    "Silhouette Score": [
        kmeans_score,
        hierarchical_score
    ]
})
comparison

small_sample = tfidf_matrix[:100].toarray()
linked = linkage(
    small_sample,
    method="ward"
)
plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("News Articles")
plt.ylabel("Distance")
plt.show()

# Conclusion

- The News Category Dataset from Kaggle was used for the project.
- TF-IDF was used to convert news text into numerical features.
- K-Means clustering grouped similar news articles.
- Hierarchical clustering was also applied for comparison.
- Silhouette Score was used to evaluate the clustering quality.
- SVD was used to visualize the high-dimensional text clusters.
- The project demonstrates how unsupervised learning can automatically group similar news articles.
