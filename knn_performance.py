from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from eda_pre_model import X, y

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
                     random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Test set predictions: n {}'.format(y_pred))
print(knn.score(X_test, y_test))