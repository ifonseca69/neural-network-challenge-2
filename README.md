### Neural Network Challenge 2- Attrition Prediction 

You are tasked with creating a neural network that HR can use to predict whether employees are likely to leave the company. Additionally, HR believes that some employees may be better suited to other departments, so you are also asked to predict the department that best fits each employee. These two columns should be predicted using a branched neural network.

**Employee attrition** refers to the gradual reduction of a workforce by employees leaving the organization for various reasons. Predicting attrition is crucial for organizations to retain valuable talent and maintain a stable workforce. Neural networks, a subset of machine learning, are particularly effective in this domain due to their ability to model complex patterns and relationships in data.

# Summary

In the provided space below, briefly answer the following questions.

1. Is accuracy the best metric to use on this data? Why or why not?

2. What activation functions did you choose for your output layers, and why?

3. Can you name a few ways that this model might be improved?

1. -  Accuracy might not always be the best metric, especially if the dataset is imbalanced. For example, if there are significantly more non-attrition cases than attrition cases, a model that predicts "no attrition" for all instances would still have high accuracy but would be useless. In such cases, metrics like precision, recall, F1-score, or the area under the ROC curve (AUC-ROC) can provide a better understanding of the model's performance.
   
2. -   **Department Output Layer**: I used the **softmax** activation function because it is suitable for multi-class classification problems. It converts the output into a probability distribution over the classes, ensuring that the sum of probabilities is 1.
    
-   **Attrition Output Layer**: I used the **sigmoid** activation function because it is suitable for binary classification problems. It outputs a probability value between 0 and 1, indicating the likelihood of attrition.

3. -   **Feature Engineering**: Create new features that might be relevant to attrition and department prediction, such as interaction terms or aggregated features.
    
-   **Hyperparameter Tuning**: Experiment with different hyperparameters such as learning rate, batch size, and number of epochs to find the optimal settings.
    
-   **Model Architecture**: Try different model architectures, such as adding more layers, increasing the number of neurons, or using different activation functions.
    
-   **Regularization**: Use techniques like dropout, L1/L2 regularization to prevent overfitting.
    
-   **Ensemble Methods**: Combine predictions from multiple models using techniques like bagging, boosting, stacking, or voting to improve accuracy and robustness.
    
-   **Cross-Validation**: Use cross-validation to ensure that the model generalizes well to unseen data and to identify if the model is overfitting or underfitting.

### Instructions

Open the starter file in Google Colab and complete the following steps, which are divided into three parts:

#### Part 1: Preprocessing

1.  Import the data and view the first five rows.
    
2.  Determine the number of unique values in each column.
    
3.  Create  `y_df`  with the attrition and department columns.
    
4.  Create a list of at least 10 column names to use as  `X`  data. You can choose any 10 columns you’d like EXCEPT the attrition and department columns.
    
5.  Create  `X_df`  using your selected columns.
    
6.  Show the data types for  `X_df`.
    
7.  Split the data into training and testing sets.
    
8.  Convert your  `X`  data to numeric data types however you see fit. Add new code cells as necessary. Make sure to fit any encoders to the training data, and then transform both the training and testing data.
    
9.  Create a StandardScaler, fit the scaler to the training data, and then transform both the training and testing data.
    
10.  Create a OneHotEncoder for the department column, then fit the encoder to the training data and use it to transform both the training and testing data.
    
11.  Create a OneHotEncoder for the attrition column, then fit the encoder to the training data and use it to transform both the training and testing data.
    

#### Part 2: Create, Compile, and Train the Model

1.  Find the number of columns in the  `X`  training data.
    
2.  Create the input layer. Do NOT use a sequential model, as there will be two branched output layers.
    
3.  Create at least two shared layers.
    
4.  Create a branch to predict the department target column. Use one hidden layer and one output layer.
    
5.  Create a branch to predict the attrition target column. Use one hidden layer and one output layer.
    
6.  Create the model.
    
7.  Compile the model.
    
8.  Summarize the model.
    
9.  Train the model using the preprocessed data.
    
10.  Evaluate the model with the testing data.
    
11.  Print the accuracy for both department and attrition.
    

#### Part 3: Summary

Briefly answer the following questions in the space provided:

1.  Is accuracy the best metric to use on this data? Why or why not?
    
2.  What activation functions did you choose for your output layers, and why?
    
3.  Can you name a few ways that this model could be improved?