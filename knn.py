import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler 

data = np.array([
    [1.0, 1.2, 'A'],
    [1.5, 1.8, 'A'],
    [5.0, 8.0, 'B'],
    [8.0, 8.8, 'B'],
    [1.1, 1.3, 'A'],
])

def predict_knn(new_point, data, k):
    """
    Predicts the class of a new_point using the K-Nearest Neighbors algorithm.
    Includes normalization (bonus) before calculating distances.
    """
    
    X = data[:, :-1].astype(float)
    y = data[:, -1]
    
    # 1. Scaling (Normalization/Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_point_scaled = scaler.transform(np.array([new_point]))[0] 
    
    scaled_data = np.column_stack((X_scaled, y))

    # 2. Calculate distances (Euclidean)
    distances = []
    for row in scaled_data:
        features = row[:-1].astype(float)
        distance = np.linalg.norm(features - new_point_scaled) 
        distances.append((distance, row[-1])) # (distance, class_label)

    # 3. Select k nearest neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # 4. Return the predicted class (majority vote)
    neighbor_classes = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(neighbor_classes).most_common(1)
    
    return most_common[0][0]

# Example Execution
new_point = [3.0, 3.5]
k = 3
predicted_class = predict_knn(new_point, data, k)

print(f"New Point: {new_point}, K: {k}")
print(f"Predicted Class: {predicted_class}")