import tensorflow as tf
print(tf.__version__)

# 加载LeNet模型
#model = tf.keras.models.load_model('lenet_model.h5')
model = tf.keras.models.load_model('lenet_model.h5')

model.summary()

import cv2
import matplotlib.pyplot as plt

# plt中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = cv2.imread('9.png')
print(img.shape)
# 原始图片
plt.title('原始图片')
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
plt.imshow(img, cmap="Greys")
plt.title('灰度')
plt.show()

# 底色和字的颜色取反
img = cv2.bitwise_not(img)
plt.imshow(img, cmap='Greys')
plt.title('底色字体颜色取反')
plt.show()

# 底色变为纯白色，字变为纯黑色
img[ img <= 144 ] = 0
img[ img > 140 ] = 255 # 130
plt.imshow(img, cmap='Greys')
plt.title('底色纯白字体纯黑')
plt.show()

img = cv2.resize(img, (32, 32))
plt.title('更改大小')
plt.show()

img = img.astype('float32')
# 数据正则化
img /= 255
plt.title('正则化')
plt.show()

# 增加维度
img = img.reshape(1, 32, 32, 1)
print(img.shape)

pred = model.predict(img)
print(pred)
# 输出结果
print(pred.argmax())
