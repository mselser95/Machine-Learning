"""
Trader modelado con Q-Learning
"""
import random
from collections import deque

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Trader:
    """
    Clase para modelar un Trader.
    Parámetros:
        state_size : Cantidad de ruedas a utilizar en la estrategia de trading.
    """
    def __init__(self, state_size):
        self.state_size = state_size  # Cantidad de ruedas a considerar en el Trading Rule
        self.action_size = 3  # Hold, Buy, Sell
        self.memory = deque(maxlen=1000)  # Jornadas que recuerda
        self.book = []  # Posición en el activo que posee

        self.gamma = 0.95  # Tasa de descuento / impaciencia
        self.epsilon = 1.0  # Tasa de 'curiosidad' / Exploration Rate
        self.epsilon_min = 0.01  # Mínimo umbral de 'curiosidad'
        self.epsilon_decay = 0.995  # Ritmo de disminusión de la 'curiosidad'

        # Trading rule
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        self.model = model

    def act(self, state):
        """
        Realiza toma la decisión.
        Parámetros:
            state : Representa la información de la jornada de que se dispone
                    para decidir.
        Retorna:
            tuple ('acción a seguir', 'si se decidió con convicción')
        """
        if random.random() <= self.epsilon:
            return (random.randrange(self.action_size), False)

        views_strength = self.model.predict(state)
        return (np.argmax(views_strength[0]), True)

    def learn(self, batch_size):
        """
        Realiza proceso de aprendizaje por Q-Learning.
        ver : https://en.wikipedia.org/wiki/Q-learning
        Parámetros:
            batch_size : Cantidad de jornadas previas de trading sobre las que
                         va a revisar su entendimiento.
        """
        ml = len(self.memory)
        mini_batch = (self.memory[i] for i in range(ml - batch_size + 1, ml))

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Al incrementar el conocimiento bajo el umbral de 'curiosidad'
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
