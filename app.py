# Import the Pandas library, which is used to work with data frames.
import pandas as pd
import streamlit as st

# Streamlit setup
st.set_page_config(layout="centered")
st.title("🌺 Iris Dataset - ML Algorithm Comparison")

# Read the 'iris.csv' file to create a data frame (DataFrame).
df = pd.read_csv('IRIS (1).csv')

# Separate independent variables (X) and the dependent variable (y) from the data frame.
# Independent variables (Features)
X = df[['sepal_length', 'sepal_width', 
        'petal_length', 'petal_width']].values

# Dependent variable (Target class)
y = df['species'].values
st.write(f"X shape: {X.shape}")
st.write(f"y shape: {y.shape}")

# Use the train_test_split function from the sklearn library to split the dataset into training and testing subsets.
from sklearn.model_selection import train_test_split

# When splitting the data into training and testing sets, specify the size of the test set and the seed for random data splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the StandardScaler class from the sklearn library to scale the data.
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object.
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)  
X_test_scaled = sc.transform(X_test)

# Import the Logistic Regression class.
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model and specify the random seed for randomness.
log_reg = LogisticRegression(random_state=0)

# Train the model using X_train and y_train.
log_reg.fit(X_train, y_train)

# Use the trained model to predict X_test data.
from sklearn.metrics import accuracy_score
y_pred_logr = log_reg.predict(X_test)
st.write(f"**Logistic Regression Accuracy:** {accuracy_score(y_test, y_pred_logr):.2%}")
st.write(f"First 5 predictions: {y_pred_logr[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Import the confusion_matrix library to calculate the confusion matrix.
from sklearn.metrics import confusion_matrix

# Calculate a confusion matrix between predictions and actual values.
cm_logr = confusion_matrix(y_test, y_pred_logr)

# Print the confusion matrix.
st.write("**Logistic Regression Confusion Matrix:**")
st.write(cm_logr)

# Import the Support Vector Classifier (SVC) class.
from sklearn.svm import SVC

# Create an SVC model with an RBF (Radial Basis Function) kernel.
svc = SVC(kernel='rbf')

# Train the model using X_train and y_train.
svc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_svc = svc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_svc = confusion_matrix(y_test, y_pred_svc)

#print accuracy
st.write(f"**SVM Accuracy:** {accuracy_score(y_test, y_pred_svc):.2%}")
st.write(f"First 5 predictions: {y_pred_svc[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Print the confusion matrix.
st.write("**SVM Confusion Matrix:**")
st.write(cm_svc)

# Import the K-Nearest Neighbors class.
from sklearn.neighbors import KNeighborsClassifier

# Create a K-Nearest Neighbors model with a specified number of neighbors (n_neighbors) and using the Minkowski distance metric.
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

# Train the model using x_train and y_train.
knn.fit(X_train, y_train)

# Use the trained model to predict x_test data.
y_pred_knn = knn.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_knn = confusion_matrix(y_test, y_pred_knn)

st.write(f"**KNN Accuracy:** {accuracy_score(y_test, y_pred_knn):.2%}")
st.write(f"First 5 predictions: {y_pred_knn[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Print the confusion matrix.
st.write("**KNN Confusion Matrix:**")
st.write(cm_knn)

# Import the Gaussian Naive Bayes class.
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes model.
gnb = GaussianNB()

# Train the model using X_train and y_train.
gnb.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_gnb = gnb.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

st.write(f"**Naive Bayes Accuracy:** {accuracy_score(y_test, y_pred_gnb):.2%}")
st.write(f"First 5 predictions: {y_pred_gnb[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Print the confusion matrix.
st.write("**Naive Bayes Confusion Matrix:**")
st.write(cm_gnb)

# Import the Decision Tree class.
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree model with entropy as the criterion.
dtc = DecisionTreeClassifier(criterion='entropy')

# Train the model using X_train and y_train.
dtc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_dtc = dtc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_dtc = confusion_matrix(y_test, y_pred_dtc)

# Print the confusion matrix.
st.write("**Decision Tree Confusion Matrix:**")
st.write(cm_dtc)

#compute accuracy for DEcision Tree classifier
st.write(f"**Decision Tree Accuracy:** {accuracy_score(y_test, y_pred_dtc):.2%}")
st.write(f"First 5 predictions: {y_pred_dtc[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Import the Random Forest class.
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest model with entropy as the criterion and 10 estimators.
rfc = RandomForestClassifier(criterion='entropy', n_estimators=10)

# Train the model using X_train and y_train.
rfc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_rfc = rfc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
st.write(f"**Random Forest Accuracy:** {accuracy_score(y_test, y_pred_rfc):.2%}")
st.write(f"First 5 predictions: {y_pred_rfc[:5]}")
st.write(f"First 5 actual: {y_test[:5]}")

# Print the confusion matrix.
st.write("**Random Forest Confusion Matrix:**")
st.write(cm_rfc)