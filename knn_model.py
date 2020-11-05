from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from eda_pre_model import iris

knn = KNeighborsClassifier(n_neighbors=6)

# 2 arguments: features and labels/targets - this returns
# the classifier itself and modifies it to fit to the data
knn.fit(iris[‘data’], iris[’target’])

# To predict unlabelled data
X_new = np.array([[5.6, 2.8, 3.9, 1.1],
                  [5.7, 2.6, 3.8, 1.3],
                  [4.7, 3.2, 1.3, 0.2]])

prediction = knn.predict(X_new)

print(X_new.shape)
print('Prediction:{}’.format(prediction))

