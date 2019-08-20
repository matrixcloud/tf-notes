import tensorflow as tf
import forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVG_DECAY = 0.99
MODEL_SAVE_PATH = './models'
MODEL_NAME = 'mnist_model'

def backward(mnist):
    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVG_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as session:
        init_op = tf.global_variables_initializer()
        session.run(init_op)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = session.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(session, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
