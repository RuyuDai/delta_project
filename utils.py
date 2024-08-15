import numpy as np

def calculate_accuracy(predicted_sentiment, true_sentiment):
    # Ensure both arrays are numpy arrays
    predicted = np.array(predicted_sentiment)
    true = np.array(true_sentiment)
    
    # Check if the arrays have the same shape
    if predicted.shape != true.shape:
        raise ValueError("The predicted and true sentiment arrays must have the same shape.")
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(predicted == true)
    
    # Calculate the total number of predictions
    total_predictions = len(true)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    return accuracy