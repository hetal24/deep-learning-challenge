# Alphabet Soup Charity: Deep Learning Model

## **Overview of the Analysis**

* The purpose of this analysis is to develop a binary classification model using deep learning to predict whether an applicant receiving funding from Alphabet Soup will be successful. Using historical data, we built and optimized a neural network model to identify patterns that could help the organization make informed funding decisions.


# Result

## Data PreProcessing 

* Target Variable:

    * IS_SUCCESSFUL

* Feature Variables:

    * APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, etc.

* Removes Variables:

    * EIN and NAME


## Compiling, Training, and Evaluating the Model

# Neural Network Structure:

* Input Layer: 
    
    * Hidden Layers:

        * Layer 1: 80 neurons, activation function = ReLU

        * Layer 2: 30 neurons, activation function = ReLU 
    
    * Output Layer:
    
        * 1 neuron with Sigmoid activation function.

# Training Details:

* Loss Function: Binary Crossentropy

* Optimizer: Adam

* Epochs: 100

* Batch Size:32

# Model Performance:

* Testing Accuracy: 72.39%

* Loss: 0.5695

# Optimization Attempts

* Adjusting the number of neurons and layers

* Changing activation functions

* Modifying the number of epochs

## Summary and Recommendations

* Our best model achieved 72.13% accuracy, which (did/did not) meet the 75% target.

* If we were to further optimize this model, we could:

    * Increase training data by gathering more applicant records.
    * Use a different model such as Random Forest or Gradient Boosting, which may handle categorical variables better.
    * Tune hyperparameters further using GridSearchCV or RandomizedSearchCV.