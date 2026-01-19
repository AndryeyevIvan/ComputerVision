import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures2.csv')

print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, y, epochs=200, verbose=0)

plt.plot(history.history['loss'], label='Loss (втрата)')
plt.plot(history.history['accuracy'], label='Accuracy (точність)')
plt.xlabel('Epoch (епоха)')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

test = np.array([[16, 16, 0]])

pred = model.predict(test)


print("\nЙмовірності для кожного класу:", pred)
print("Модель визначила:", encoder.inverse_transform([np.argmax(pred)]))