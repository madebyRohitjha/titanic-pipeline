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

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
print (model)

from sklearn.pipeline import Pipeline

pipeline = Pipeline ([
    ("preprocessor", preprocessor),
    ("model", model)
])

print (pipeline)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline,X_train, y_train, cv=5)

print(scores)
print("average:", scores.mean()) 

rf = pipeline.named_steps["model"]

print(rf.feature_importances_)

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

print(feature_names)