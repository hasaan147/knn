import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title('K-Nearest Neighbors Classification with Iris Dataset')

    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    st.subheader('Iris Dataset Overview')
    st.write("### Dataset Preview")
    st.write(df.head())

    # Split the data into features and target variable
    X = df.drop('species', axis=1)
    y = df['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sidebar for KNN parameters
    st.sidebar.header('KNN Parameters')
    k = st.sidebar.slider('Number of Neighbors (k)', 1, 20, 5)

    # Initialize and fit KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = knn.predict(X_test_scaled)

    # Evaluate the model
    st.subheader('Model Evaluation')
    st.write(f"### Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("### Classification Report:\n" + classification_report(y_test, y_pred))

    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Find the optimal number of neighbors using cross-validation
    st.subheader('Optimal Number of Neighbors')
    k_range = range(1, 21)
    mean_scores = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())

    # Plot the cross-validation results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, mean_scores, marker='o')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Mean Cross-Validated Accuracy')
    ax.set_title('Optimal Number of Neighbors')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
