import math

def sigmoid(z):
    """
    Implements the sigmoid (logistic) function.
    """
    return 1 / (1 + math.exp(-z)) # [cite: 36, 37]

# Linear model output (z) from the features and weights
z = 1.72 

# Calculate the probability
probability = sigmoid(z) # [cite: 39]

# Classification Decision (If probability >= 0.5, classify as 1, otherwise 0)
classification = 1 if probability >= 0.5 else 0

print(f"Linear Model Output (z): {z}")
print(f"Calculated Probability: {probability:.4f}")
print(f"Classification Decision (Threshold 0.5): Class {classification}")