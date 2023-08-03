import tensorflow as tf
import keras
import numpy as np
import datetime, os

from keras.backend import set_session

def set_seed():
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42) 
    tf.config.experimental.enable_op_determinism()

def tensorboard_callback():
    """
    Creates checkpoints and saves them in a /logs folder under the date of creation.

    Returns a TensorBoard callback
    """
    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  
    return tf.keras.callbacks.TensorBoard(logdir)

def close_session():
    tf.keras.backend.clear_session()

def score_model(predictions, true_values):
    """
    This function takes network's predictions and the true_values associated with the test set.
    It turns the probabilities of labels into actual 0 and 1 values, which are appended to a list in the correct order.
    Next it creates a list of true labels and in the end scores each predicted label against the correct one.

    returns 'score' that is accuracy
    """
    
    # Let's begin by defining the variables that we will use
    predicted_labels = []
    true_labels = []
    correct_predictions = 0

    # First step will go through every predicted array and turn them into labels 0 or 1
    for prediction in predictions:
        # We have to go to index 0, since the probability is first item in an array
        if prediction[0] < 0.5:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
    
    # Next step is going through the dataframe and append every true label to a dictionary
    for label in true_values:
        true_labels.append(label)
    
    # Now we will count the amount of correct predictions
    # To do that we will go through both predicted labels and true labels, one of both at a time and put them agains each other 
    # If the predictions is correct the amount of correct predictions increases by 1 
    for x in range(len(true_labels)):
        if predicted_labels[x] == true_labels[x]:
            correct_predictions += 1
        else:
            pass
    
    # Finally we divide the number of correct predictions by the number of all labels in the test set
    score = correct_predictions/len(true_values)

    return score 
    
def sequencer(input_shape, units1, units2, rate1, iteration, units3=None, rate2=None, rate3=None):
    sequence=[]
    suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    layers = tf.keras.layers
    if iteration == 3:
        sequence.append(layers.BatchNormalization(input_shape= input_shape, name=f"IL_{suffix}")),
        sequence.append(layers.Dense(units= units1, activation= "relu", name=f"HDL1_{suffix}")),
        sequence.append(layers.Dense(units= units2, activation= "relu", name=f"HDL2_{suffix}")),
        sequence.append(layers.Dropout(rate= rate1, name=f"HDpL_{suffix}")),
        sequence.append(layers.Dense(units= 1, activation= "sigmoid", name=f"OL_{suffix}")),
    else:
        sequence.append(layers.BatchNormalization(input_shape= input_shape, name=f"IL_{suffix}")),
        sequence.append(layers.Dense(units= units1, activation= "relu", name=f"HDL1_{suffix}")),
        sequence.append(layers.Dropout(rate= rate1, name=f"HDpL1_{suffix}")),
        sequence.append(layers.BatchNormalization(name=f"HBNL_{suffix}")),
        sequence.append(layers.Dense(units= units2, activation= "relu", name=f"HDL2_{suffix}")),
        sequence.append(layers.Dropout(rate= rate2, name=f"HDpL2_{suffix}")),
        sequence.append(layers.Dense(units= units3, activation= "relu", name=f"HDL3_{suffix}")),
        sequence.append(layers.Dropout(rate= rate3, name=f"HDpL3_{suffix}")),
        sequence.append(layers.Dense(units= 1, activation= "sigmoid", name=f"OL_{suffix}"))

    return sequence

def network_maker(input_shape, units1, units2, rate1, iteration, units3=None, rate2=None, rate3=None):
    set_seed()
    if iteration == 3:
        sequence = sequencer(input_shape, units1, units2, rate1, iteration)
    else:
        sequence = sequencer(input_shape, units1, units2, rate1, iteration, units3, rate2, rate3)
    new_model = keras.Sequential(sequence)
    new_model.compile(
        optimizer= "adam",
        loss= "binary_crossentropy",
        metrics= ["binary_accuracy"]
    )

    return new_model

def network_dict(input_shape, params_list, callbacks, data, iteration):
    net_idx = 0
    network_dict = {}
    X_train, X_test, y_train, y_test = data
    print(f"Scoring model {net_idx + 1}/{len(params_list)}")
    for param_set in params_list:
        set_seed()
        close_session()
        if iteration == 3:
            units1, units2, rate1 = param_set
            new_model = network_maker(input_shape, units1, units2, rate1, iteration)
        else:
            units1, units2, units3, rate1, rate2, rate3 = param_set
            new_model = network_maker(input_shape, units1, units2, rate1, iteration, units3, rate2, rate3)
        
        new_model.fit(
            x= X_train,
            y = y_train,
            batch_size= 20,
            epochs= 200,
            callbacks= callbacks,
            verbose= False
        )
            
        new_score = score_model(
            predictions= new_model.predict(
                x= X_test,
                batch_size= 20,
                verbose= True
            ),
            true_values= y_test
        ) * 100

        net_idx += 1
        if net_idx < (len(params_list)):
            print(f"Scoring model {net_idx+1}/{len(params_list)}")
        else:
            print("Done scoring!")
        network_dict[f"Model_{net_idx}"] = [new_model, new_score]
        
    return network_dict

def save_network(model, name):
    """
    This function takes the model and a name it should be saved under.
    Model gets saved in a .h5 format file in the projects main folder.
    """
    suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"./{name}_{suffix}.h5"
    model.save(path)
        



