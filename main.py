from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.read_csv("train.csv")

#print(df.head())
#print(df.info())
X= df.drop(["Survived","PassengerId","Name","Ticket","Cabin"],axis=1)

y = df["Survived"]

#print(X.head())
#print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state= 42)

#print(X_test.shape)
#print(y_test.shape)

numerical_cols = X.select_dtypes(
    include=["int64","float64"]
).columns

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

print("numerical_columns",numerical_cols)

print("categorical_coulmns",categorical_cols)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline= Pipeline([("imputer", SimpleImputer(strategy="median"))])

print(num_pipeline)

cat_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy ="most_frequent")),
    ("encoder",OneHotEncoder())
    ])

print(cat_pipeline)

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ("num",num_pipeline, numerical_cols),
    ("cat",cat_pipeline,categorical_cols)

])

print(preprocessor)
