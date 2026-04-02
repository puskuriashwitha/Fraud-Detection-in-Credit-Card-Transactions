import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("creditcard.csv")

# Convert Time to Hour
data["Hour"] = (data["Time"] // 3600) % 24

# Create many locations for training
locations = [
"Hyderabad","Chennai","Bangalore","Mumbai","Delhi",
"Kolkata","Pune","Dubai","New York","London",
"Paris","Singapore","Tokyo","Sydney","Toronto",
"Berlin","Rome","Madrid","Los Angeles","San Francisco"
]

np.random.seed(42)
data["Location"] = np.random.choice(locations, len(data))

# Encode location names
encoder = LabelEncoder()
data["LocationCode"] = encoder.fit_transform(data["Location"])

# Features
X = data[["Hour","Amount","LocationCode"]]
y = data["Class"]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

# Save model + encoder
pickle.dump(model,open("fraud_model.pkl","wb"))
pickle.dump(encoder,open("location_encoder.pkl","wb"))

print("Model trained successfully")