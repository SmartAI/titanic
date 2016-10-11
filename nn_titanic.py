import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
sns.set_style('whitegrid')


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()

train_df.info()
print '-'*80
test_df.info()

train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)

train_df['Embarked'] = train_df['Embarked'].fillna('S')
print train_df['Embarked']

# plot
sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, ax=axis2)
embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S','C','Q'], ax=axis3)

embark_dummies_train = pd.get_dummies(train_df['Embarked'])
print embark_dummies_train
# drop S in Embarked dummies
embark_dummies_train.drop(['S'], axis=1, inplace=True)
embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)
train_df = train_df.join(embark_dummies_train)
test_df = test_df.join(embark_dummies_test)
train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

# fill na value in Test of Fare
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)
train_df.info()
print '-' * 40
test_df.info()
print test_df['Fare']

# Fare hist
train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, xlim=(0,50))

# Fare vs Survived
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
fare_avg = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
std_fare.index.names = fare_avg.index.names = ['Survived']
fare_avg.plot(kind='bar', yerr=std_fare, legend=False)

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15,4))
axis1.set_title('Age original value')
axis2.set_title('New age values')
# age in train
train_age_avg = train_df['Age'].mean()
train_age_std = train_df['Age'].std()
train_count_age_nan = train_df['Age'].isnull().sum()
# age in test
test_age_avg = test_df['Age'].mean()
test_age_std = test_df['Age'].std()
test_count_age_nan = test_df['Age'].isnull().sum()

# generate new ages in range [mean-3*std, mean + 3*std]
rand_1 = np.random.randint(train_age_avg - 3 * train_age_std,
                           train_age_avg + 3 * train_age_std,
                           train_count_age_nan)
rand_2 = np.random.randint(test_age_avg - 3 * test_age_std,
                           test_age_avg + 3 * test_age_std,
                           test_count_age_nan)
# plot original age hist
train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# fill nan values
train_df['Age'][np.isnan(train_df['Age'])] = rand_1
test_df['Age'][np.isnan(test_df['Age'])] = rand_2
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)

# plot new age hist
train_df['Age'].hist(bins=70, ax=axis2)

# kde plot of age vs survived
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()

age_avg = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
fig, axis = plt.subplots(1,1,figsize=(18,4))
sns.barplot(x='Age', y='Survived', data=age_avg, ax=axis)

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

# Family
# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

train_df = train_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)

# No need to use Sex column since we created Person column
train_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child', 'Female', 'Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)

train_df.drop('Person', axis=1, inplace=True)
test_df.drop('Person', axis=1, inplace=True)

sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_df,size=5)

pclass_dummies_train  = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_df.drop('Pclass', axis=1, inplace=True)
test_df.drop('Pclass', axis=1, inplace=True)

train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)

train_df.info()

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

## using tensorflow   TF lear
import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder as feeder

learning_rate = 0.0001
training_epochs = 2000
batch_size = 32
display_step = 1

X = X_train.values.astype(np.float32)
Y = Y_train.values.astype(np.float32)

X_train = X[:600]
Y_train = Y[:600]
X_test = X[600:]
Y_test = Y[600:]

dataiter = feeder.setup_train_data_feeder(x=X_train, y=Y_train, n_classes=2, batch_size=batch_size)
x, y = dataiter.input_builder()
feed_dict_fun = dataiter.get_feed_dict_fn()

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 128 # 2nd layer number of features
n_hidden_3 = 64 # 2nd layer number of features
n_input = 9 # data input (feature shape: 9)
n_classes = 2 # one-hot encoded output


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# define network
net = multilayer_perceptron(x, weights, biases)

# define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y))


# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = Y_train.shape[0]/batch_size
        for i in range(total_batch):
            _,c = sess.run([optimizer, cost], feed_dict=feed_dict_fun())
            avg_cost += c / total_batch
        print 'Epoch[{}], cost: {}'.format(epoch+1, avg_cost)

        # define accuracy
    correct = tf.equal(tf.argmax(net, 1), np.asarray(Y_test, dtype=np.int64))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))
    print 'Accuracy : {}'.format(acc.eval({x:X_test}))



    X_test = test_df.drop('PassengerId', axis=1).copy().values.astype(np.float32)
    Y_pred = tf.argmax(net, 1)

    submission = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred.eval({x:X_test})})
    submission.to_csv('titanic_nn.csv', index=False)
