Documentation

- used one hot encoding to deal with categorical data
- there is a big imbalance, only ~4% 'Active' labels
- trained some models
- did downsampling / upsampling
- Results are good on test split of training data but low on submission (0.25 best) -> Evaluated on balanced test set so the result was warped

Results

upsampling
F1_score of Multilayer Perceptron Classifier 0.2073029569609597
F1_score of Support Vector Classifier 0.22415080253826053
F1_score of Linear Support Vector Classifier 0.21559028431557142

downsampling
F1_score of Multilayer Perceptron Classifier 0.21673499813641445
F1_score of Support Vector Classifier 0.22370523545446003
F1_score of Linear Support Vector Classifier 0.21655447046553525

Ideas for next steps:
- try dealing with imbalance by weighting the cost in the support vector classifier
- incorporate the order of the values
- look at loss vs. epoch and gauge wether overfitting or not

Tried new way of one hot encoding by encoding every character of the sequence separately. Performance increasers dramtically:
F1_score of Multilayer Perceptron Classifier downsampling 0.65... --> 0.65.. public score
F1_score of Multilayer Perceptron Classifier upsampling 0.88414 --> 0.8929 public score

Try to train longer and with more training data (90% train 10% test)
MLPClassifier(random_state=random_state, max_iter=1000, verbose=1,
                                                          hidden_layer_sizes=1000, tol=0.000001)

