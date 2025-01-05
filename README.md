# Machine_learning_Supervised_Learning

## **Project: Predictive Modeling with Classification and Clustering**

### **Overview**
This project aims to perform end-to-end data analysis, from preprocessing and exploration to building predictive models, with both unsupervised and supervised learning techniques. The dataset includes multiple features that describe various attributes of transactions, such as **Quantity**, **Value**, **Weight**, **Category**, and **Payment Terms**, and the goal is to predict the target variable, a classification label with three classes (0, 1, 2). The project explores both clustering (unsupervised learning) and classification (supervised learning) approaches to build models that can accurately predict the target class.

### **Key Phases**

#### **1. Data Preprocessing**
- **Data Cleaning**: Identified and handled missing values, duplicates, and irrelevant data. 
- **Feature Engineering**: Transformed categorical features using encoding techniques, scaled numerical features, and created new relevant features.
- **Data Splitting**: The dataset was divided into training and test sets to evaluate model performance effectively.

#### **2. Exploratory Data Analysis (EDA)**
- **Descriptive Statistics**: Basic summary statistics were computed for numerical features to understand their distributions and relationships.
- **Data Visualization**: Visualizations such as histograms, box plots, and correlation matrices were used to explore feature distributions and identify patterns in the data.
- **Outlier Detection**: Identified and handled outliers in numerical features to ensure model robustness.

#### **3. Unsupervised Learning: Clustering**
- **Clustering Algorithms**: Applied K-Means and Agglomerative Clustering to uncover underlying patterns in the data.
- **Evaluation Metrics**: Used Silhouette Score and Davies-Bouldin Index to evaluate the quality of the clusters formed.
- **Cluster Analysis**: Investigated the distribution of data points within clusters to gain insights into natural groupings within the dataset.

#### **4. Supervised Learning: Classification**
- **Models Applied**: Trained three different supervised models—**Decision Tree**, **Logistic Regression**, and **Random Forest**—to predict the target class.
- **Model Evaluation**: Assessed model performance using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Model Comparison**: The models were compared to determine the best-performing model in terms of classification accuracy and other evaluation metrics.

### **Objectives**
- To preprocess the raw dataset by cleaning and transforming it into a suitable format for modeling.
- To perform EDA and understand the relationships between features and their distributions.
- To apply unsupervised learning techniques (Clustering) and evaluate cluster quality.
- To train and evaluate supervised learning models (Decision Tree, Logistic Regression, and Random Forest) for classification.
- To compare model performance and select the best model for real-world applications.

### **Results and Insights**
- **Clustering Results**: The K-Means and Agglomerative Clustering algorithms were able to identify distinct clusters in the dataset. The Silhouette Score and Davies-Bouldin Index were used to evaluate clustering quality, with K-Means performing slightly better in terms of separation between clusters.
- **Classification Results**: Logistic Regression outperformed both Decision Tree and Random Forest in terms of accuracy, with an accuracy of **0.34**. The Decision Tree model provided good interpretability but had a similar performance to Random Forest, which performed slightly worse.
- **Model Insights**: 
  - **Logistic Regression** demonstrated the best overall classification performance and is the recommended model for deployment.
  - **Decision Tree** provided a more interpretable model, which could be useful in situations requiring explainability.
  - **Random Forest** offered a slight improvement over the Decision Tree in terms of generalization but did not outperform Logistic Regression.

### **Key Findings**
- Logistic Regression is the most effective classification model for this dataset, offering the highest accuracy and the best overall performance.
- Clustering methods revealed natural groupings in the data that could potentially provide insights into market segmentation or customer behavior.
- The decision to use unsupervised learning helped identify valuable patterns that could guide future feature engineering or business strategy.

### **Recommendations**
- **Model Deployment**: Deploy **Logistic Regression** for the classification task as it performs better than other models.
- **Further Analysis**: Explore additional features, especially categorical features, and apply advanced feature engineering techniques.
- **Model Tuning**: Tune hyperparameters for the models to improve performance further. Grid search and cross-validation can be applied for better results.
- **Cluster Interpretation**: Use the clustering results to gain business insights. The identified clusters could be used for segmentation or to tailor business strategies.

### **Technologies Used**
- **Python**: Programming language used for data analysis, preprocessing, and model development.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Matplotlib / Seaborn**: Data visualization libraries for EDA.
- **Scikit-learn**: Machine learning library used for classification, clustering, and evaluation metrics.
- **Jupyter Notebooks**: Environment for developing and testing the models.
