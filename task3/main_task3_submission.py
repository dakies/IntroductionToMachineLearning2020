import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import f1_score
from sklearn.utils import resample

# Reproducability
random_state = 4
np.random.seed(random_state)

"""
    -- IML Project Task 3 Code by Daniel Kiesewalter and Leiv Andresen --
    
    - Task description:
     
    The task is to classify antibody proteins into active or inactive
    states based on mutation information provided in the form of a string of four letters 
    corresponding to four amino acides. Due to class imbalance the score is evaluated using 
    the F1 metric.
    
    - Approach description: 
    
    Initial data exploration shows that there are only around 4% active samples in the
    training data so we decided to implement both upsampling of minority class and downsampling
    of majority class strategies to handle the class imbalance. Upsampling the minority class
    gave better results on the validation set (10% of the training data), probably due to the 
    higher amount of information as compared to the "wasteful" downsampling.
    
    In order to feed the categorical data into a model we implemented one hot encoding. This ensures
    equal distance in feature space between features with differing amino acids and hence does not
    introduce biases. Each letter was encoded separately.
    
    For the classification we used a Multilayer Perceptron as it gave better initial results when compared
    to Supportvector-based classifiers. We tuned the hyperparameters on the validation set reaching
    satisfying results with 1000 hidden layers and the default 'relu' activation and 'adam' solver.
    
    - Execution requirements:
    
    Tested only on Python version 3.7
    modules: sklearn, pandas, numpy
    train.csv and test.csv have to be located in the same folder as the script.
    The script will save the one hot encoded features for training and testing as .csv 
    and reload them for future execution to reduce runtime.
    
    """


def one_hot_encoding(df):
    print("One hot encoding ...")
    # Split string into chars
    df_chars = pd.DataFrame([list(x['Sequence']) for index, x in df.iterrows()])
    acid_types = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                  'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

    # One hot encoding with sklearn
    enc = OneHotEncoder(sparse=False)
    df_one_hot = pd.DataFrame(enc.fit_transform(df_chars))
    print("Done.")
    return df_one_hot, enc


def handle_imbalance(x, y, type_='downsample'):
    # Add labels to data
    data_ = x
    data_.insert(0, "Active", y)

    # Separate majority and minority classes
    df_majority = data_[data_['Active'] == 0]
    df_minority = data_[data_['Active'] == 1]

    if type_ is 'downsample':
        # Upsample minority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample with replacement
                                           n_samples=len(df_minority.index),  # to match majority class
                                           random_state=random_state)  # reproducible results
        # Combine majority class with upsampled minority class
        df_resampled = pd.concat([df_minority, df_majority_downsampled])
    elif type_ is 'upsample':
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority.index),  # to match majority class
                                         random_state=random_state)  # reproducible results
        # Combine majority class with upsampled minority class
        df_resampled = pd.concat([df_majority, df_minority_upsampled])
    else:
        print("[handle_imbalance] Wrong type specified.")
        return

    # Separate x and y
    y_balanced = df_resampled['Active']
    x_balanced = df_resampled.drop('Active', axis=1)

    # Display new class counts
    print("Resampled with strategy: ", type_, "Value counts: ", y_balanced.value_counts())
    return x_balanced, y_balanced


if __name__ == '__main__':
    # Load data
    training_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Try to load one hot encoded data from disk or create if it doesn't exist
    try:
        # Load one hot encoded data, # iloc to get rid of the index columns
        X_submission = pd.read_csv('X_submission_one_hot_extensive.csv').iloc[:, 1:]
        x_training = pd.read_csv('X_training_one_hot_extensive.csv').iloc[:, 1:]
        print("Successfully loaded one_hot encoded feature data.")
    except FileNotFoundError:
        # One hot encoding
        X_submission, enc_test = one_hot_encoding(test_data)
        x_training, enc_train = one_hot_encoding(training_data)

        # Check the categories are the same after one hot encoding
        for index in range(len(enc_test.categories_)):
            if all(enc_train.categories_[index] == enc_test.categories_[index]):
                pass
            else:
                print("ERROR: One hot encoding not consistent!")

        # Save one hot encoded values
        X_submission.to_csv("X_submission_one_hot_extensive.csv")
        x_training.to_csv("X_training_one_hot_extensive.csv")

    y_training = training_data['Active']

    # Train test split, stratify to ensure equal distribution of classes
    X_train, X_test, y_train, y_test = train_test_split(
        x_training, y_training, test_size=0.10, random_state=random_state, stratify=y_training)

    # Dealing with imbalance fro training data
    X_train, y_train = handle_imbalance(X_train, y_train, type_='upsample')

    # Instantiate models
    print("Fitting models")
    models = {
        "Multilayer Perceptron Classifier": MLPClassifier(random_state=random_state, max_iter=1000, verbose=1,
                                                          hidden_layer_sizes=1000, tol=0.000001)
        # "Support Vector Classifier": SVC(verbose=1, random_state=random_state),
        # "Linear Support Vector Classifier": LinearSVC(verbose=1, random_state=random_state)
    }

    # Train models
    for model_name, model in models.items():
        print("Training", model_name, "...")
        model.fit(X_train, y_train)

    # Evaluate
    for model_name, model in models.items():
        y_test_hat = model.predict(X_test)
        print("F1_score of", model_name, f1_score(y_true=y_test, y_pred=y_test_hat))

    # Predict on the test set and save to .csv
    y_submission = models["Multilayer Perceptron Classifier"].predict(X_submission)
    np.savetxt('Y_submission_MLP_upsampled_final_changed_rand_seed.csv', y_submission, delimiter=",", fmt='%d')
