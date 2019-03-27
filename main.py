import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


def readData():
    # read train set data
    df_train = pd.read_csv('train.csv')
    df_train['train'] = 1
    # read test set data
    df_test = pd.read_csv('test.csv')
    df_test['train'] = 0
    df = pd.concat([df_train, df_test], sort=False,
                   ignore_index=True)  # concatenate the training set data and test set data
    return df


def onehot(df, col):
    new_df = pd.get_dummies(df[col], prefix=col, sparse=True)
    return new_df


def trainMlp(x_data, y_data, x_test, y_test):
    # split the train set and validation set
    train_data, val_data, train_label, val_label = train_test_split(x_data, y_data, test_size=0.1, random_state=10)

    print((x_test.shape))
    print((y_test.shape))
    print((train_data.shape))
    print((train_label.shape))
    # set the dimension for hidden layer
    H_DIM = 4

    # number of features in train dataset
    col = len(train_data[0])

    # initialize tf
    x = tf.placeholder(float, [None, col])
    y = tf.placeholder(float, [None, 1])

    # initialize input layer
    w1 = (tf.Variable(tf.random_normal([col, H_DIM])))
    b1 = tf.Variable(tf.constant(0.0, shape=[H_DIM]))
    # w2 = tf.Variable(tf.random_normal([H_DIM, H_DIM]))
    # b2 = tf.Variable(tf.constant(0.0, shape=[H_DIM]))

    # initialize the output layer
    w3 = tf.Variable(tf.random_normal([H_DIM, 1]))
    b3 = tf.Variable(tf.constant(0.0, shape=[1]))
    hidden_layer = tf.nn.tanh(tf.matmul(x, w1) + b1)

    # hidden_layer2 = tf.nn.tanh(tf.matmul(hidden_layer, w2) + b2)
    yhat = (tf.matmul(hidden_layer, w3) + b3)

    loss = tf.reduce_mean(tf.abs((yhat) - y))  # use mean average error as training loss
    loss2 = tf.reduce_mean(tf.square(tf.abs(yhat) - y))  # use mean square error as validation loss
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)  # use Adam optimizer

    init = tf.global_variables_initializer()  # initialize all the data
    sess = tf.Session()
    sess.run(init)

    # train 400,000 steps
    for step in range(0, 20000):
        sess.run(optimizer, feed_dict={x: train_data, y: train_label})
        if (step % 400 == 0):
            # observe the loss2 for each session, pick the best loss2, write the csv
            print(sess.run(loss2, feed_dict={x: train_data, y: train_label}))
            print(sess.run(loss2, feed_dict={x: val_data, y: val_label}))
            # print(sess.run(loss2, feed_dict={x: x_test, y: y_test}))

            # write the ans csv for testset
    ans_y = sess.run(yhat, feed_dict={x: x_test, y: train_label})
    ans = pd.DataFrame(columns=['Id', 'time'])
    ans['Id'] = [i for i in range(0, 100)]
    ans['time'] = [(abs(i[0])) for i in ans_y]
    ans.to_csv('submission.csv', index=0)


def getCleanData(orgin_df):
    df = pd.DataFrame(orgin_df)

    # do onehot for 'penalty'
    df = df.merge(onehot(orgin_df, 'penalty'), left_index=True, right_index=True)

    # n_jobs for x=-1
    def fun(x):
        if (x == -1):
            return 16
        else:
            return x

    df['n_jobs'] = df['n_jobs'].apply(lambda x: fun(x))
    return df


def doFeatureEngineeriing(df):
    df['new_feature'] = np.log(df.max_iter * df.n_samples * df.n_features * df.n_classes / (1 + df.n_jobs))
    #df['time'] = np.log(df['time'])
    # normalization features
    columns = df.columns.tolist();
    columns.remove('id')
    columns.remove('penalty')
    columns.remove('time')
    columns.remove('train')
    columns.remove('random_state')
    columns.remove('scale')
    columns.remove('l1_ratio')
    columns.remove('n_clusters_per_class')
    columns.remove('flip_y')
    columns.remove('n_informative')
    for feature in columns:
        df[feature] = df[[feature]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_train = df[df['train'] == 1]
    df_test = df[df['train'] == 0]

    x_train = df_train[columns]
    y_train = df_train[['time']]
    x_test = df_test[columns]
    y_test = df_test['time']

    # convert to np.array format
    x_data = np.array(x_train, dtype=np.float32)
    y_data = np.array(y_train, dtype=np.float32)

    x_test_data = np.array(x_test, dtype=np.float32)
    y_test_data = np.array(y_test, dtype=np.float32)

    return x_data, y_data, x_test_data, y_test_data


if __name__ == '__main__':
    orgin_df = readData()
    df = getCleanData(orgin_df)
    x_data, y_data, x_test_data, y_test_data = doFeatureEngineeriing(df)
    trainMlp(x_data, y_data, x_test_data, y_test_data.reshape(100, 1))
