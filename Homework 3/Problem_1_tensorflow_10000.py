import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential  #removed python from each layer
from keras import layers
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical

training_10000 = np.genfromtxt('./data/training_10000.csv', delimiter=',')
testing_100000 = np.genfromtxt('./data/test_100000.csv', delimiter=',')

training_10000_x = training_10000[:, :3]
training_10000_y = training_10000[:, 3:].astype('int')
testing_100000_x = testing_100000[:, :3]
testing_100000_y = testing_100000[:, 3:].astype('int')

# One-hot encode the targets
training_10000_y = to_categorical(training_10000_y, num_classes=4)
testing_100000_y = to_categorical(testing_100000_y, num_classes=4)

inputs = training_10000_x.astype('float32')
targets = training_10000_y.astype('int')
val_inputs = testing_100000_x.astype('float32')
val_targets = testing_100000_y.astype('int') 


# Model configuration
fold_no = 1

acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=10, shuffle=True)

fold_predictions = []

for train, test in kfold.split(inputs, targets):
    model = Sequential([
        layers.Dense(64, input_dim=3, activation='relu'), 
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
            optimizer='adam', 
            metrics=['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    print('------------------------------------------------------------------------')


    history = model.fit(inputs[train], targets[train], 
                        validation_data=(val_inputs, val_targets), 
                        batch_size=64,
                        epochs=50,
                        verbose=2)
    
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    predictions = model.predict(inputs)
    fold_predictions.append(np.array(predictions))
    

print('Average Acc per Fold: ' + str(sum(acc_per_fold)/len(acc_per_fold)))
print('Average Loss per Fold: ' + str(sum(loss_per_fold)/len(loss_per_fold)))

final_predictions = sum(fold_predictions) / 10

np.savetxt('./predictions/predictions_10000.csv', final_predictions, delimiter=',')

# 80.07999956607819
# 0.47866792380809786
