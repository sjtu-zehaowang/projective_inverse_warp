import tensorflow as tf
import transform
import os
import cv2
tf.compat.v1.disable_eager_execution()

def main():
    i = 0
    directory_name = "E:/pycharm_project/transform/image"
    for filename in os.listdir(directory_name):
        i = i+1
        src_img = cv2.imread(directory_name + "/" + filename)
        src_number = filename[0:6]
        src_number = int(src_number)

        step = 3
        tgt_img = transform.compose(transform.tgt_img(src_img = src_img, src_number = src_number, step = step))
        tgt_img = tgt_img.transform
        sess = tf.compat.v1.Session()
        tgt_img = sess.run(tgt_img)
        print(tgt_img.shape)
        Img_Name = "E:/pycharm_project/transform/results/tgt_image" + str(i) + ".png"
        cv2.imwrite(Img_Name, tgt_img)

main()