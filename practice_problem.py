# def fib(n):
    # if n == 0:
        # return 0
    # elif n == 1 :
        # return 1
    # else:
        # return fib(n-1) + fib(n-2)
    # 
# print(fib(7))

import numpy as np
import cv2

image = cv2.imread('/home/easemyai/Downloads/cat_or_dog_2.jpg')

labels = ['dog', 'cat', 'panda']

np.random.seed(1)

w = np.random.rand(3, 3072)
b = np.random.rand(3)

resize_img = cv2.resize(image, (32, 32)).flatten()

scores = w.dot(resize_img) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}:{:.2f}".format(label, score))

cv2.putText(image, "Label:{}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


