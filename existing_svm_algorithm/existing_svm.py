"""
1. Pre-processed text data is inputted. (Problem 1) 
2. Assign the input vector to an entity space with a complex dimensional feature. (Problem 3)
3. Find the best hyperplane position that maximizes the margin between instances of different classes, effectively splitting the feature space into regions corresponding to different classes. 
4. For handling nonlinear classification tasks, SVM maps the feature space from the input space to a higher-dimensional space using kernel tricks. (Problem 3) 
5. Use Multi-Class Classification Strategies [One-vs-All (OVA) or One-vs-One (OVO)] for multi-class classification.
6. SVM computes the position of a new instance relative to the decision boundary. (Problem 1)
7. SVM assigns class labels based on their position as a training result. (Problem 2)
8. Performance will be evaluated with the common four indicators of accuracy rate, precision rate, recall rate and F1 value.
"""

#data preprocessing
