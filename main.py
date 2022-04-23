import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")


file_location = "covtype.csv"
df_original = pd.read_csv(file_location)

# print(df_original.head())
# print(df_original.shape)
# print(len(df_original))

# class distribution

sns.countplot(x="Cover_Type", data=df_original)


# Another way to see the class distribution
df_original.groupby("Cover_Type").size()

# Creating train and test data
df_features = df_original.drop(columns=["Cover_Type"])
df_class_label = df_original[["Cover_Type"]]
df_class_label = np.ravel(df_class_label)

# Usual train and test split

X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_class_label, test_size=0.33, random_state=42
)

# Fitting the model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

"""
Hyperparameter setting: (kept it default)

RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
"""

model_predict = model_rf.predict(X_test)
print(f" Accuracy: {metrics.accuracy_score(y_test, model_predict)}")


"""
If you wanna use the model to predict a single instance
we have to expand the dimension or what the model expects does not match
with the dimension of the instance
I am just taking a single instance from the test data
"""
a_instance = np.expand_dims(X_test.iloc[2], 0)
# print(a_instance.shape)

model_predict = model_rf.predict(a_instance)

# small function to test if the instance is equal to the original label or not.
def test_instance(test, original):
    return True if test == original else False


# print(test_instance(model_predict, y_test[2]))
