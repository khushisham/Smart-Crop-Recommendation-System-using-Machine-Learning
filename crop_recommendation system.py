
**Importing Required libraires**
"""

import pandas as pd # data analysis
import numpy as np # linear algebra

"""**Dataset**

"""

crop = pd.read_csv("/content/Crop_recommendation.csv")
crop

"""**Data Exploration and Inspection**"""

crop.shape

crop.info()

crop.isnull().sum()

crop.duplicated().sum()

crop.describe()

"""**Data Type Conversion and Correlation Analysis**"""

# Convert the 'label' column to a categorical data type.
crop['label'] = crop['label'].astype('category')

# Calculate the correlation matrix only for numeric columns.
corr = crop.corr(numeric_only=True)
corr

"""**Visualization**"""

import seaborn as sns
sns.heatmap(corr,annot=True, cbar=True, cmap='coolwarm' )

crop['label'].value_counts()

import matplotlib.pyplot as plt
sns.distplot(crop['N'])
plt.show()

"""**Categorical Encoding and Frequency Analysis of Crops**"""

crop_dict = {
    'rice' : 1,
    'maize' : 24,
    'jute' : 3,
    'cotton' : 4,
    'coconut' : 5,
    'papaya': 6,
    'orange' : 7,
    'apple' : 8,
    'muskmelon' : 9,
    'watermelon' : 10,
    'grapes' : 11,
    'mango' : 12,
    'banana' : 13,
    'pomegranate' : 14,
    'lentil' : 15,
    'blackgram' : 16,
    'mungbean' : 17,
    'mothbeans' : 18,
    'pigeonpeas' : 19,
    'kidneybeans' : 20,
    'chickpea' : 21,
    'coffee' : 22,
}
crop['crop_num'] = crop['label'].map(crop_dict).astype(int)
crop['crop_num'].value_counts()

crop.shape

x = crop.drop(['crop_num','label'],axis=1)
y = crop['crop_num']
x.shape

y.shape

"""**Splitting data into training and testing set**"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape

x_test.shape

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train

import pandas as pd # Import the pandas library

# Convert x_train and x_test to Pandas DataFrames
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

# Now you can use select_dtypes
x_train = x_train.select_dtypes(include=['number'])
x_test = x_test.select_dtypes(include=['number'])

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)
x_train

y_train

"""**Importing Machine Learning Models and Evaluation Metrics**"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import multilabel_confusion_matrix

def multiclass_accuracy_score(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    return accuracy.mean()

models = {
    'Logistic Regression' : LogisticRegression(),
    'Naive Bayes' : GaussianNB(),
    'Support Vector Machine' : SVC(),
    'K-Nearest Neighbors' : KNeighborsClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Bagging' : BaggingClassifier(),
    'AdaBoost' : AdaBoostClassifier(),
    'Gradient Boosting' : GradientBoostingClassifier(),
    'Extra Trees' : ExtraTreeClassifier(),
}

print(x_train.shape)
for name, md in models.items():
    md.fit(x_train,y_train)
    ypred = md.predict(x_test)

    print(f"{name}  with accuracy : {multiclass_accuracy_score(y_test,ypred)}")

"""**Selecting Model**"""

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
ypred = rfc.predict(x_test)
accuracy_score(y_test,ypred)

x_train[0]

rfc.predict([x_train[0]])

x_train[0]

"""**Crop Recommendation Function Using Random Forest Classifier**"""

def recommendation(x_train):

    features = np.array([x_train[0]])

    prediction = rfc.predict(features).reshape(1,-1)



    return prediction[0]

x_train[0],y_train[0]

N = 48
P = 45

k = 30

temperature = 40.77463

humidity = 70.413269

ph = 80.78006

rainfall = 400.774507

predict = recommendation([x_train[3]])

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",

                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",

                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",

                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

if predict[0] in crop_dict:

    crop = crop_dict[predict[0]]

    print("{} is a best crop to be cultivated ".format(crop))

else:

    print("Sorry we are not able to recommend a proper crop for this environment")

