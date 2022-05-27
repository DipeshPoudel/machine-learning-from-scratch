from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf_model = KNN(k=4)
clf_model.fit(X_train, y)
y_pred = clf_model.predict(X_test)
print(y_pred)
