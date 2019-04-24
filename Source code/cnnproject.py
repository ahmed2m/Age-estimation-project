import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import cv2
import pickle as pk
import tensorflow as tf

# dir = "UTKFace/"
# filenames = os.listdir(dir)
# filenames = filenames[:23708]
# lst = [dir + f for f in filenames]
# # arr=np.array(lst)
# data = []
# age_label = []
# classes = []
# labels = []
# targetreg = []
# race_label = []
# gender_label = []
# print(filenames[0])
# print(lst[0])
#
# t = zip(lst, filenames, list(range(len(filenames))))
# # print(len(list(t)))
#
# a = datetime.datetime.now().replace(microsecond=0)
# for f, source, i in t:
#     img = cv2.imread(f)
#     # SCALING
#     scale = 40
#     dim = (scale, scale)
#     img = cv2.resize(img, dim)
#
#     # Grayscaling
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     sp = source.split("_")
#     age = sp[0]
#     gender = sp[1]
#     race = sp[2]
#     flat = img.flatten()
#     if "jpg" in race:
#         print("race: " + race + "\nsource:" + source)
#         continue
#     row = np.array([[age, gender, race]])
#     age = int(age)
#     targetreg.append(age)
#     if age <= 14:
#         labels.append([1, 0, 0, 0, 0])
#     if (age > 14) and (age <= 25):
#         labels.append([0, 1, 0, 0, 0])
#     if (age > 25) and (age < 40):
#         labels.append([0, 0, 1, 0, 0])
#     if (age >= 40) and (age < 60):
#         labels.append([0, 0, 0, 1, 0])
#     if age >= 60:
#         labels.append([0, 0, 0, 0, 1])
#
#     gender_label.append(gender)
#     race_label.append(race)
#     age_label.append(age)
#     data.append(flat)
#
# b = datetime.datetime.now().replace(microsecond=0)
# print("Finished import in " + str(b - a))
# data = np.asarray(data)
# labels = np.asarray(labels)
# print(data.shape)

# pick_out = open('data.pickle', 'wb')
# pk.dump(data, pick_out)
# pick_out.close()
#
# pick_out = open('labels.pickle', 'wb')
# pk.dump(labels, pick_out)
# pick_out.close()


pick_in = open('data.pickle', 'rb')
data= pk.load(pick_in)
pick_in.close()

pick_in = open('labels.pickle', 'rb')
labels = pk.load(pick_in)
pick_in.close()

n_classes = 5
batch_size = 1000

x = tf.placeholder('float', [None, 1600])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=32)


def convolution_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([10 * 10 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    #     weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    #                'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    #                'W_conv3': tf.Variable(tf.random_normal([5, , 64, 128])),
    #                'W_fc': tf.Variable(tf.random_normal([13 * 13 * 128, 128])),
    #                'out': tf.Variable(tf.random_normal([128, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              # 'b_conv3': tf.Variable(tf.random_normal([128])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 40, 40, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    #     conv3 = conv2d(conv2,weights['W_conv3']) + biases['b_conv3']
    #     conv3 = maxpool2d(conv3)

    #     fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.reshape(conv2, [-1, 10 * 10 * 64])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


saver = tf.train.Saver()


def train_neural_network(x):
    prediction = convolution_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            tr_acc_list = []
            epoch_loss = 0
            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size

                x_batch = X_train[start:end]
                y_batch = y_train[start:end]
                # print("label: ", epoch_y[1])
                # print("shape image: ",epoch_x.shape ,'\nlabel : ', epoch_y.shape)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})

                epoch_loss += c
                #               tr_acc_list.append(train_acc)

                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #         print('train Accuracy:', accuracy.eval({x: X_train, y: y_train}))
        acc = accuracy.eval({x: X_test, y: y_test})

        print('Test Accuracy:', accuracy.eval({x: X_test, y: y_test}))

        save_path = saver.save(sess, "cnnmodel.ckpt")
        pk_out = open('acc.pickle', 'wb')
        pk.dump(acc, pk_out)
        pk_out.close()


train_neural_network(x)


def Cnn_predict(image):
    prediction = convolution_neural_network(x)
    img = cv2.resize(image, (40, 40))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = []
    flat = img.flatten()
    data.append(flat)
    data = np.asarray(data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "cnnmodel.ckpt")

        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: data}), 1))
        print(result[0])




