import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# set display options for easy viewing while using IDE or otherwise
pandas.options.display.width = 1000
pandas.options.display.max_rows = 1000
pandas.options.display.max_columns = 1000

df = pandas.read_csv('Iris.csv')
species = df['Species'].unique()
# create a dictionary of integers 0 to n mapped to unique label values in this case species
d = dict(zip(species, [i for i in range(len(species))]))
label = df['Species'].map(d)
# replace columns species values with respective integer values
df.drop(columns='Species', inplace=True)
df['Species'] = label
X = df.iloc[:, :5]
y = df['Species'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = RandomForestClassifier()
classifier.fit(X, y)
y_predicted = classifier.predict(X_test)
print(accuracy_score(y_test, y_predicted))
