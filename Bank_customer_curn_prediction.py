import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from google.colab import files

# Load Dataset
dataset = pd.read_excel("/content/churn_modelling_10000_rows-1.xlsx")

# Input and Output
input_data = dataset.iloc[:, :-1]
output_data = dataset.iloc[:, -1]

# Feature Scaling
ss = StandardScaler()
input_data = pd.DataFrame(
    ss.fit_transform(input_data),
    columns=input_data.columns
)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    input_data,
    output_data,
    test_size=0.2,
    random_state=42
)

# Build ANN Model
ann = Sequential()
ann.add(Dense(32, input_dim=8, activation="relu"))
ann.add(Dense(16, activation="relu"))
ann.add(Dense(8, activation="relu"))
ann.add(Dense(4, activation="relu"))
ann.add(Dense(1, activation="sigmoid"))

# Compile Model
ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
ann.fit(x_train, y_train, batch_size=100, epochs=100)

# Training Accuracy
train_predictions = ann.predict(x_train)

pred_data1 = []
for i in train_predictions:
    if i[0] > 0.5:
        pred_data1.append(1)
    else:
        pred_data1.append(0)

print("Training Accuracy:", accuracy_score(y_train, pred_data1) * 100)

# Testing Prediction
test_predictions = ann.predict(x_test)

pred_data = []
for i in test_predictions:
    if i[0] > 0.5:
        pred_data.append(1)
        print("Bank chhodega")
    else:
        pred_data.append(0)
        print("Bank nahi chhodega")

print("Testing Accuracy:", accuracy_score(y_test, pred_data) * 100)

print("Bank chhodne wale customers:", pred_data.count(1))
print("Bank me rahne wale customers:", pred_data.count(0))
print("Total customers:", len(pred_data))

# Upload New Excel File
uploaded = files.upload()

# Read New File
new_data = pd.read_excel(list(uploaded.keys())[0])

# Remove Target Column if Present
if "Exited" in new_data.columns:
    new_data = new_data.drop(columns=["Exited"])

# Scale New Data
new_data = pd.DataFrame(
    ss.transform(new_data),
    columns=new_data.columns
)

# Predict on New Data
new_predictions = ann.predict(new_data)

new_pred = []

for i in new_predictions:
    if i[0] > 0.5:
        new_pred.append(1)
        print("Bank chhodega")
    else:
        new_pred.append(0)
        print("Bank nahi chhodega")

print("New Data - Bank chhodne wale:", new_pred.count(1))
print("New Data - Bank me rahne wale:", new_pred.count(0))
print("New Data - Total Customers:", len(new_pred))