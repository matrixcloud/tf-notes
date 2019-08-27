###############################
# 模型测试
###############################
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import mnist_backward
import mnist_generateds

TEST_INTERVAL_SECS = 10
TEST_NUM = 10000

def test():
    with tf.Graph().as_default() as g:
        # 占位
        x  = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])

        # 定义前向传播，预测y
        y = forward.forward(x, None)
        # 实例化还原滑动平均的saver
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVG_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        prev_step = None

        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, is_train=False)

        while True:
            with tf.Session() as session:
                # 加载已经训练好的模型
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                # 如果有 checkpoint 模型则恢复
                if ckpt and ckpt.model_checkpoint_path:
                    # 恢复会话
                    saver.restore(session, ckpt.model_checkpoint_path)
                    # 恢复轮数
                    print('model_checkpoint_path', ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    finished = True if prev_step == global_step else False
                    prev_step = global_step
                    if (finished):
                        return
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=session, coord=coord)
                    # 计算准确率
                    xs, ys = session.run([img_batch, label_batch])
                    accuracy_score = session.run(accuracy, feed_dict={x: xs, y_: ys})
                    print('After %s training step(s), test accuracy=%g' % (global_step, accuracy_score))

                    coord.request_stop()
                    coord.join(threads)
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(TEST_INTERVAL_SECS)


if __name__ == "__main__":
    test()