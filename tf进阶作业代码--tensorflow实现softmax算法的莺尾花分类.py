import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def batches(batch_size, features, labels):
    """
    实现获取批量的方法
    :param batch_size:  批量大小
    :param features:    输入的特征数据集
    :param labels:      输入数据的标签
    :return:
    """
    assert len(features) == len(labels)
    output_batches = []

    # 循环迭代获取批量数据。
    for start_idx in range(0, len(features), batch_size):
        end_idx = start_idx + batch_size
        batch = [features[start_idx: end_idx], labels[start_idx: end_idx]]
        output_batches.append(batch)
    return output_batches


# tf.set_random_seed(2018)
iris_feature=[u'花萼长度',u'花萼宽度',u'花瓣长度',u'花瓣宽度']
path='datas/iris.data'
data=pd.read_csv(path,header=None)
x_prime=data[list(range(4))]
# y=pd.Categorical(data[4]).codes.reshape(-1,1)
# print(x_prime,y)
y=pd.get_dummies(data[4])
# final_data=pd.concat([x_prime,y],axis=1)
# print(final_data)

with tf.Graph().as_default():
	#设置模型超参
	#数据的特征维度
	n_input=4
	#标签的类别(没有做one-hot)
	n_classes=3
	#学习率
	lr=0.01
	#迭代次数
	epochs=40000
	#
	batch_size=150

	input_x=tf.placeholder(tf.float32,shape=[None,n_input],name='x')
	input_y=tf.placeholder(tf.float32,shape=[None,n_classes],name='y')

	weights=tf.get_variable('w',shape=[n_input,n_classes],dtype=tf.float32,
	                        initializer=tf.random_normal_initializer(stddev=0.1))
	biases=tf.get_variable('b',shape=[n_classes],dtype=tf.float32,initializer=tf.zeros_initializer())

	#进行正向传播，获取logits和预测值
	logits=tf.add(tf.matmul(input_x,weights),biases)
	pred=tf.nn.softmax(logits)

	#构建模型损失
	with tf.name_scope('loss'):
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=input_y,
			logits=logits
		))
		#定义一个可视化输出操作
		tf.summary.scalar('train_loss',tensor=loss)

	#构建模型优化器
	optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
	train_opt=optimizer.minimize(loss)

	#计算模型准确率
	with tf.name_scope('accuracy'):
		correct_pred=tf.equal(tf.argmax(logits,axis=1),
		                      tf.argmax(input_y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
		tf.summary.scalar('acc',tensor=accuracy)

	#创建一个持久化对象
	saver=tf.train.Saver()
	with tf.Session() as sess:
		#变量初始化
		sess.run(tf.global_variables_initializer())
		#实现可视化操作
		summary=tf.summary.merge_all()
		writer=tf.summary.FileWriter('./xianyu',graph=sess.graph)

		#数据分训练集和测试集
		X_train,Y_train=x_prime,y
		X_test,Y_test=x_prime,y
		batch=batches(batch_size,X_train,Y_train)

		# print('-------------------')
		# print(batch_test)
# from sklearn.model_selection import train_test_split
# 		X_train, X_test, Y_train, Y_test = train_test_split(x_prime, y, test_size=0.3, random_state=0)
		step = 1
		for e in range(epochs):
			for trainx,trainy in batch:
				# print(trainx,trainy)
				train_dict={input_x:trainx,input_y:trainy}
				sess.run(train_opt,train_dict)
				if step%10==0:
					train_loss,train_acc,summary_=sess.run([loss,accuracy,summary],train_dict)
					print('Epoch:{}-Step:{}-Train Loss:{:.5f}-Train Acc:{:.4f}'.format(e,step,train_loss,train_acc))
					writer.add_summary(summary_,global_step=step)
					#持久化操作

				# if step%20==0:
				# 	valid_dict = {input_x: testx, input_y: testy}
				# 	valid_loss, valid_acc = sess.run([loss, accuracy], valid_dict)
				# 	print('Epochs:{}-Step:{}-Valid Loss:{:.5f}-Valid Acc:{:.4f}'.format(e, step, valid_loss, valid_dict))
				step+=1
		#验证集
		print('weight,biaes:',sess.run([weights,biases]),sep='\n')
		save_path='./model'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
			print('成功创建持久化路径：{}'.format(save_path))

		batch_test = batches(batch_size, X_test, Y_test)
		valid_dict = {input_x: batch_test[0][0], input_y: batch_test[0][1]}
		#valid_dict = {input_x: X_test, input_y: Y_test}
		valid_loss, valid_acc = sess.run([loss, accuracy], valid_dict)
		print('Valid Loss:{:.5f}-Valid Acc:{:.4f}'.format(valid_loss,valid_acc))
		# print('Valid Loss:{:.5f}-Valid Acc:{:.4f}'.format(valid_loss, valid_dict))

		file_name='model.cpkt'
		save_file=os.path.join(save_path,file_name)
		saver.save(sess=sess,save_path=save_file)
		print('成功将模型保存到路径:{}'.format(save_file))

		#绘图
		# y = pd.Categorical(data[4]).codes[:batch_size]
		# y_true=data[4][:batch_size]
		y_true=pd.Categorical(data[4]).codes[:batch_size]
		# y_hat=tf.nn.softmax(logits)
		# y_hat=pred
		# print('y_true:',y_true)

		y_hat_one_hot = sess.run(pred, feed_dict={input_x: batch_test[0][0]})
		# print('pred',pred)

		y_hat=tf.argmax(y_hat_one_hot,1)

		# print('y_hat:',sess.run(y_hat))

		x_test_len=np.arange(len(batch_test[0][0]))
		plt.figure(figsize=(12,9),facecolor='w')
		plt.ylim(-1,3)
		plt.plot(x_test_len,y_true,'ro',markersize=6,zorder=3,label=u'true value')
		plt.plot(x_test_len,sess.run(y_hat),'yo',markersize=16,zorder=1,label=u'pred value，acc=%.4f'%valid_acc)
		plt.legend(loc='upper right')
		plt.xlabel(u'data number',fontsize=18)
		plt.ylabel(u'kinds',fontsize=18)
		plt.title(u'flower kinds',fontsize=20)
		plt.show()






