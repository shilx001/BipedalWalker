import numpy as np
import tensorflow as tf

################## hyper parameters ##################

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.99
TAU = 0.001
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
HIDDEN_SIZE = 256
REPLAY_START = 10000
DROP_OUT_PROBABILITY = 0.85


################## DDPG algorithm ##################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float64)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.replay_start = REPLAY_START
        self.S = tf.placeholder(tf.float64, [None, s_dim], name='S')
        self.S_ = tf.placeholder(tf.float64, [None, s_dim], name='S_')
        self.R = tf.placeholder(tf.float64, [None, ], name='reward')
        self.a = self._build_a(self.S)
        self.done = tf.placeholder(tf.float64, [None, ], name='done')
        q = self._build_c(self.S, self.a)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q, 这里我改为reduce_sum
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_ * (1 - self.done)
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def scale(self, state):
        # 对state进行放缩
        return (state - self.s_mean) / self.s_std

    def choose_action(self, s):
        s = np.reshape(s, [-1, self.s_dim])
        if self.pointer > self.replay_start:
            s = self.scale(s)
        action=self.sess.run(self.a, feed_dict={self.S: s})
        return action

    def learn(self):
        if self.pointer < self.replay_start:
            return
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        if self.pointer == self.replay_start:  # data normalization
            # 对experience replay的数据进行处理，找出需要放缩的mean和std
            # 主要问题在于每个特征都要放缩
            states = self.memory[:self.replay_start, :self.s_dim]
            self.s_mean = np.mean(states, axis=0)
            self.s_std = np.std(states, axis=0)

        bt = self.memory[indices, :]  # transitions data
        bs = self.scale(bt[:, :self.s_dim])
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim]
        bs_ = self.scale(bt[:, -self.s_dim - 1:-1])
        bdone = bt[:, -1]
        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: bdone})

    def store_transition(self, s, a, r, s_, done):  # 每次存储一个即可
        s = np.reshape(np.array(s), [self.s_dim, 1])
        a = np.reshape(np.array(a), [self.a_dim, 1])
        r = np.reshape(np.array(r), [1, 1])
        s_ = np.reshape(np.array(s_), [self.s_dim, 1])
        done = np.reshape(np.array(done), [1, 1])

        transition = np.vstack((s, a, r, s_, done))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = np.reshape(transition, [2 * self.s_dim + self.a_dim + 1 + 1, ])
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable) + tf.random_normal(
                [1, HIDDEN_SIZE], dtype=tf.float64, stddev=0.2)
            h1 = tf.nn.dropout(h1, keep_prob=DROP_OUT_PROBABILITY)
            h1 = tf.contrib.layers.layer_norm(h1)
            h2 = tf.layers.dense(h1, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable) + tf.random_normal(
                [1, HIDDEN_SIZE], dtype=tf.float64, stddev=0.2)
            h2 = tf.nn.dropout(h2, keep_prob=DROP_OUT_PROBABILITY)
            h2 = tf.contrib.layers.layer_norm(h2)
            h3 = tf.layers.dense(h2, units=self.a_dim, activation=tf.nn.tanh, trainable=trainable)
            return h3 * self.a_bound

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            input_s = tf.reshape(s, [-1, self.s_dim])
            input_a = tf.reshape(a, [-1, self.a_dim])
            input_all = tf.concat([input_s, input_a], axis=1)  # s: [batch_size, s_dim]
            h1 = tf.layers.dense(input_all, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable)
            h1 = tf.nn.dropout(h1, keep_prob=DROP_OUT_PROBABILITY)
            h2 = tf.layers.dense(h1, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable)
            h2 = tf.nn.dropout(h2, keep_prob=DROP_OUT_PROBABILITY)
            h3 = tf.layers.dense(h2, units=1, activation=None, trainable=trainable)
            return h3
