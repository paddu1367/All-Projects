import pytest
import numpy as np
from Test_NN_custom_class import CNN
import os


def test_append_conv2d_layer_maxpooling2d_layer_flatten_layer_dense_layer():
    input = np.zeros((20,32, 32, 1))
    model = CNN()
    model.add_input_layer(shape=(32, 32, 1), name="input0")
    model.append_conv2d_layer(6, (5, 5),padding='valid', activation='tanh',name="conv2d_layer1")
    out = model.predict(input)
    assert (out.shape == (20,28,28,6))
    model.append_maxpooling2d_layer(pool_size=(2, 2),padding='valid', name='pooling1')
    model.append_conv2d_layer(16, (5, 5),padding='valid', activation='tanh',name="conv2d_layer2")
    out = model.predict(input)
    assert (out.shape == (20,10,10,16))
    model.append_maxpooling2d_layer(pool_size=(2, 2),padding='valid', name='pooling2')
    model.append_conv2d_layer(120, (5, 5),padding='valid', activation='tanh',name="conv2d_layer3")
    out = model.predict(input)
    assert (out.shape == (20,1,1,120))
    model.append_flatten_layer(name='flatten')
    model.append_dense_layer(84,activation="tanh",name="dense1")
    out = model.predict(input)
    assert (out.shape == (20,84))
    model.append_dense_layer(10,activation="softmax",name="dense2")
    out = model.predict(input)
    assert (out.shape == (20,10))

def test_set_training_flag_get_weights_without_biases_1():
    my_cnn = CNN()
    input_size=np.random.randint(32,100)
    number_of_dense_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=(input_size,),name="input")
    previous_nodes=input_size
    layer_list=[]
    for k in range(number_of_dense_layers):
        number_of_nodes = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_dense_layer(num_nodes=number_of_nodes,name="test_get_weights_without_biases_1"+str(k))
        actual = my_cnn.get_weights_without_biases(layer_number=k+1)
        layer_list.append(k+1)
        assert actual.shape ==  (previous_nodes,number_of_nodes)
        previous_nodes=number_of_nodes
    my_cnn.set_training_flag(layer_numbers=layer_list,trainable_flag=False)
    for k in range(number_of_dense_layers):
        assert not my_cnn.model.layers[k+1].trainable


def test_evaluate_train():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    #loading iris data 
    iris = load_iris()
    X = iris.data[:, :]
    y = iris.target
    # Split data ito training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    # Normalize input data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    my_cnn = CNN()
    my_cnn.add_input_layer(shape=(4,),name="input")
    my_cnn.append_dense_layer(3,activation="softmax",name="dense1")
    my_cnn.set_metric(['accuracy'])
    my_cnn.set_loss_function('sparse_categorical_crossentropy')
    my_cnn.set_optimizer(optimizer="SGD",learning_rate=0.1)
    history = my_cnn.train(X_train, y_train, num_epochs=10, batch_size=10)
    assert len(history)==10
    # Evaluate model
    loss, acc = my_cnn.evaluate(X_test, y_test)
    assert loss<0.4 and acc>0.8

def test_evaluate_train2():
    from tensorflow.keras.datasets import cifar10 # type: ignore
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    my_cnn=CNN()
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=3,padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=num_classes,activation="relu",name="dense1")
    my_cnn.set_metric(['accuracy'])
    my_cnn.set_loss_function('sparse_categorical_crossentropy')
    my_cnn.set_optimizer(optimizer="SGD")
    history = my_cnn.train(X_train, y_train, num_epochs=epochs, batch_size=batch_size)
    assert len(history)==epochs
    # Evaluate model
    (X_test, y_test)=(X_test[0:num_predictions, :], y_test[0:num_predictions])
    loss, acc = my_cnn.evaluate(X_test, y_test)
    assert loss!= None and acc!= None