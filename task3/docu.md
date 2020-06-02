Documentation

- used one hot encoding to deal with categorical data
- there is a big imbalance, only ~4% 'Active' labels
- trained some models
- did downsampling / upsampling
- Results are good on test split of training data but low on submission (0.25 best)

Results

upsampling
F1_score of Multilayer Perceptron Classifier 0.8484710055355512
F1_score of Support Vector Classifier 0.8342102465986394
F1_score of Linear Support Vector Classifier 0.8090899170667831

downsampling
F1_score of Multilayer Perceptron Classifier 0.816127911018422
F1_score of Support Vector Classifier 0.8256567724326168
F1_score of Linear Support Vector Classifier 0.8154059680777238

Ideas for next steps:
- try dealing with imbalance by weighting the cost in the support vector classifier
- look at loss vs. epoch and gauge wether overfitting or not