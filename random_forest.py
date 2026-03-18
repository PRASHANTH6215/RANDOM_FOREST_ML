import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    'pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi','diabetespedigreefunction','age','outcome'
]
df = pd.read_csv(url,names=columns)
print(df.head())

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum)

cols = ['pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi']
im = SimpleImputer(missing_values=np.nan,strategy='mean')
for col in cols:
  df[col] = im.fit_transform(df[col].values.reshape(-1,1))
print(df.isnull().sum)

plt.figure(figsize=(8,5))
sns.countplot(x='outcome',data=df)
plt.title("diabetes distribution")
plt.show()

df.hist(figsize=(12,10),bins = 20)
plt.suptitle("Feature distribution")
plt.show()

plt.figure(figsize=(14,10))
for i, col in enumerate(df.columns[:-1]):
  plt.subplot(3,3,i+1)
  sns.histplot(df[col], kde=True)
  plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,10))
for i, col in enumerate(df.columns[:-1]):
  plt.subplot(3,3,i+1)
  sns.boxplot(x=df[col])
  plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 -Q1
df = df[(df['bmi']>Q1 - 1.5*IQR) & (df['bmi']<Q3 + 1.5*IQR)]

df['bmi_AGE_Ratio'] = df['bmi'] / df['age']
df['glucose_bmi'] = df['glucose'] / df['bmi']
df['age_group'] = pd.cut(df['age'],bins=[20, 30, 40, 50 ,60,80],labels=[1,2,3,4,5])
df['age_group'] = df['age_group'].astype(float).fillna(99).astype(int)
x = df.drop('outcome',axis=1)
y = df['outcome']
x.columns

scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

from re import split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues' )
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

