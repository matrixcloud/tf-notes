###############################
# 模型应用
###############################
import tensorflow as tf
import numpy as np
from PIL import Image
import forward
import backward

def predict(data):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        predicted_value = tf.argmax(y, 1)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVG_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                predicted_value = session.run(predicted_value, feed_dict={x: data})
                return predicted_value
            else:
                print('No checkpoint file found')
                return -1

def read_image(path):
    img = Image.open(path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    threshold = 60
    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 - img_arr[i][j]
            if (img_arr[i][j] < threshold):
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 255
    reshaped = img_arr.reshape([1, 784])
    reshaped = reshaped.astype(np.float32)
    r = np.multiply(reshaped, 1.0/255.0)

    return r

def start():
    tests = input('Input the number of test images: ')
    for i in range(int(tests)):
        path = input('The path of test image: ')
        data = read_image(path)
        predicted_value = predict(data)
        print('The predicted number is: %s' % predicted_value)

if __name__ == "__main__":
    start()