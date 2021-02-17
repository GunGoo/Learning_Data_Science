## TensorFlow Certificate Study

**Callbacks**: To stop training at specific threshold during epoch

```
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True
...
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

**Flatten**: A layer that takes input shape square and turns into a simple linear array.

```python
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

