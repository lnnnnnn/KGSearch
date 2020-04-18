# coding=utf-8

from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras.initializers import Constant, Ones, Zeros

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
									 initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
									initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

def rerank_g(n_feats, n_types, feat_weights=None, type_weights=None):
	in_feats = Input((n_feats,), dtype="float32")
	fw_layer = Dense(1, name="g_weight")
	x = fw_layer(in_feats)
	x = LayerNormalization()(x)

	if feat_weights and len(feat_weights) == 2 and len(feat_weights[0]) == n_feats:
		fw_layer.set_weights([
			np.array(feat_weights[0]).reshape((n_feats,1)), 
			np.array(feat_weights[1]).reshape((1,))
		])

	in_type = Input((1,), dtype="int32")
	tw_layer = Embedding(n_types, 1, name="g_type")
	tw = tw_layer(in_type)

	if type_weights and len(type_weights) == n_types:
		tw_layer.set_weights([np.array(type_weights)])

	outs = Lambda(lambda x:1+x[0]*x[1][0])([x, tw])
	return [in_feats, in_type], outs

class TimeDiff(Dense):
    def call(self, inputs):
        output = 1 / (K.dot(inputs, self.kernel) + 1)
        if self.use_bias:
            output = output * self.bias + 1 - self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

def rerank_p(time_weights=[6.43e-8, 0.2]):	# time_weights = [w_t, w_i]
	in_tdif = Input((1,), dtype="float32")
	timediff = TimeDiff(1, name="p_time")
	outs = timediff(in_tdif)

	if time_weights and len(time_weights) == 2:
		timediff.set_weights([
			np.array(time_weights[0]).reshape((1,1)), 
			np.array(time_weights[1]).reshape((1,))
		])

	return [in_tdif], outs

def build_model_orig(n_feats, n_types, feat_weights=None, type_weights=None):
	bm25 = Input((1,), dtype="float32")
	intent = Input((1,), dtype="float32")
	g_i, g_s = rerank_g(n_feats, n_types, feat_weights, type_weights)
	p_i, p_s = rerank_p()

	outs = Lambda(lambda x:x[0]*x[2]*(1+x[1]*x[3]))([bm25, intent, g_s, p_s])
	inps = [bm25, intent] + g_i + p_i
	score_model = Model(inps, outs)

	inps_neg = [Input(inp.shape[1:], dtype=inp.dtype) for inp in inps]
	outs_neg = score_model(inps_neg)
	outp = Lambda(lambda x:K.sigmoid(x[0]-x[1]))([outs, outs_neg])
	#outp = Lambda(lambda x:K.sigmoid(x[0])-K.sigmoid(x[1]))([outs, outs_neg])
	loss_model = Model(inps+inps_neg, outp)
	loss_model.compile("adam", "binary_crossentropy", ["accuracy"])
	#loss_model.compile("adam", "hinge", ["accuracy"])
	return score_model, loss_model

def build_model(n_feats, n_types, n_layers=3, n_hiddens=64, dropout=0.2, feat_weights=None, type_weights=None):
	bm25 = Input((1,), dtype="float32")
	intent = Input((1,), dtype="int32")
	in_feats = Input((n_feats,), dtype="float32")
	in_type = Input((1,), dtype="int32")
	in_tdif = Input((1,), dtype="float32")
	inps = [bm25, intent, in_feats, in_type, in_tdif]

	intent_onehot = Lambda(lambda x:K.one_hot(K.cast(x[:,0], "int32"), 2))(intent)
	type_onehot = Lambda(lambda x:K.one_hot(K.cast(x[:,0], "int32"), n_types))(in_type)

	x = concatenate([bm25, intent_onehot, in_feats, type_onehot, in_tdif])
	x = LayerNormalization()(x)

	for _ in range(n_layers):
		x = Dense(n_hiddens, activation="tanh")(x)
		x = Dropout(dropout)(x)


	'''
	x: Tensor("dropout_3/cond/Merge:0", shape=(?, 64), dtype=float32)
	outs: Tensor("dense_5/BiasAdd:0", shape=(?, 1), dtype=float32)

	'''
	print('x:',x)
	outs = Dense(1)(x)
	outs = Dense(1)(Lambda(lambda x:K.concatenate(x))([outs, bm25]))#将表达式封装为Layer
	print('outs:',outs)
	score_model = Model(inps, outs) 

	inps_neg = [Input(inp.shape[1:], dtype=inp.dtype) for inp in inps]
	outs_neg = score_model(inps_neg)
	outp = Lambda(lambda x:K.sigmoid(x[0]-x[1]))([outs, outs_neg]) #预测相关性概率
	loss_model = Model(inps+inps_neg, outp)
	loss_model.compile("adam", "binary_crossentropy", ["accuracy"])
	return score_model, loss_model

if __name__ == "__main__":
	model, lmodel = build_model(9, 5)#9种从搜索日志获取的文档属性 type有5类：blog wiki article ...
	model.summary()
	lmodel.summary()