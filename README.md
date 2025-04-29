# EXNO:4-DS

### NAME: SATHYAA R
### REG NO: 212223100052

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
# FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np
```

```
df=pd.read_csv("bmi.csv")
df.head()
```

![image](https://github.com/user-attachments/assets/246c6765-cb3a-4213-be5a-7af01ffa2184)

```
df_null_sum=df.isnull().sum()
df_null_sum
```

![image](https://github.com/user-attachments/assets/cc4360b6-58c7-4b4d-a42d-74adfbc11464)

```
df.dropna()
```

![image](https://github.com/user-attachments/assets/f72fb085-16a9-4d63-bd09-60be40a0c2c8)

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

![image](https://github.com/user-attachments/assets/6576283e-561d-483c-8b0f-1bcbc6bd9d77)

```
# Standard Scaling

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```

![image](https://github.com/user-attachments/assets/23ddf3f5-b0ee-485f-9974-c7fcb3efa022)

```
sc=StandardScaler()
```
```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![image](https://github.com/user-attachments/assets/369e1fcf-8e62-44ca-8d73-42367cf43598)

```
#MIN-MAX SCALING

from sklearn.preprocessing import MinMaxScaler
```
```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/3013b116-aed5-4a3d-99ed-dbc98255d179)

```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
```
```
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/3e73f6dd-d346-415c-a575-bf9dcb502e9a)

```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
```
```
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

![image](https://github.com/user-attachments/assets/f54d41d2-49ec-4ce6-a993-660be58978fb)

```
#FEATURE SELECTION:

df=pd.read_csv("income.csv")
df.info()
```

![image](https://github.com/user-attachments/assets/ed6d1fe0-3ca2-4e9d-a425-b72b8a5fa6aa)

```
df_null_sum=df.isnull().sum()
df_null_sum
```

![image](https://github.com/user-attachments/assets/51f2db9b-73e3-4e1e-8f8a-8d2e19c58d66)

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/de771866-fc11-4d3c-a810-be3f983f2f37)

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/d9da9813-3b47-4c59-8845-9cbbb1f6ed9f)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/0ef500b1-cc9d-4f75-9ce7-86be7c35949a)

```
y_pred = rf.predict(X_test)
```

```
df=pd.read_csv("income.csv")
df.info()
```

![image](https://github.com/user-attachments/assets/f2c18e7f-e2c1-4b7f-9862-6277e53a76c0)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
```
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/afc016cc-fc59-4b9c-bf47-30345217fffb)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/f8b66daa-8a6b-4868-bc8c-b117b415025f)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

![image](https://github.com/user-attachments/assets/85b46642-4d63-417e-84e5-911c1104091c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
```
```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/789e16ac-8332-4913-81c5-ff59f586e2e1)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

![image](https://github.com/user-attachments/assets/38883e0e-e886-492a-9737-14a54569b7f7)

```
!pip install skfeature-chappers
```

![image](https://github.com/user-attachments/assets/39c5dcb9-4fde-4f78-84fe-d54f54ad1776)

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/faf2be81-e55c-4cf7-96a0-b609760ffabc)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```

```
selected_features_anova = X.columns[selector_anova.get_support()]
```

```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

![image](https://github.com/user-attachments/assets/39dc7c60-19b4-4128-8193-ba694736a617)

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```
```
df=pd.read_csv("income.csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```

```
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/5ac75327-788b-4f47-a0b0-9973843bd64d)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```
logreg = LogisticRegression()
```

```
n_features_to_select =6
```

```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![image](https://github.com/user-attachments/assets/0abb0a1e-6281-4c29-9a7b-00029e25c06a)



# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
