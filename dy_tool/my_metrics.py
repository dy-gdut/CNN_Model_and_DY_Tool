import  numpy as np
import  sklearn
from sklearn.metrics.ranking import _binary_clf_curve

def compute_binary_AP(y_true, y_score, pos_label=1):
	y_true=np.array(y_true).flatten()
	y_score=np.array(y_score).flatten()
	AP=sklearn.metrics.average_precision_score(y_true, y_score,pos_label=pos_label)
	return AP


def compute_binary_F1(y_true, y_score, pos_label=1):
	y_true=np.array(y_true).flatten()
	y_score=np.array(y_score).flatten()
	F1=sklearn.metrics.f1_score(y_true, y_score,pos_label=pos_label)
	return F1

def compute_values(y_true, y_score, pos_label=None,
						   sample_weight=None):
	"""返回最大F1值时的 TP,FP,FN,TN,threshold,F1_max
	"""
	y_true=np.array(y_true).flatten()
	y_score=np.array(y_score).flatten()
	fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
											 pos_label=pos_label,
											 sample_weight=sample_weight)
	precision = tps / (tps + fps)
	precision[np.isnan(precision)] = 0
	recall = tps /tps[-1]
	count=np.size(y_true)
	count_posivite=tps[-1]
	count_negative=count-count_posivite
	# stop when full recall attained
	# and reverse the outputs so recall is decreasing
	last_ind = tps.searchsorted(tps[-1])
	# sl = slice(last_ind, None, -1)
	# precision= np.r_[precision[sl], 1]
	# recall=np.r_[recall[sl], 0]
	# thresholds=thresholds[sl]
	#计算最大F1
	F1s = 2 * (precision * recall) / (precision + recall)
	index = np.argmax(F1s)
	F1_max=F1s[index]
	#返回最大F1值时的参数
	threshold=thresholds[index]
	TP=tps[index]
	FP=fps[index]
	FN=count_posivite-TP
	TN=count_negative-FP
	return TP,FP,FN,TN,threshold,F1_max


def compute_with_max_acc(y_true, y_score, pos_label=None,
						   sample_weight=None):
	"""返回最大F1值时的 TP,FP,FN,TN,threshold,F1_max
	"""
	y_true=np.array(y_true).flatten()
	y_score=np.array(y_score).flatten()
	fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
											 pos_label=pos_label,
											 sample_weight=sample_weight)
	precision = tps / (tps + fps)
	precision[np.isnan(precision)] = 0
	recall = tps /tps[-1]
	count=np.size(y_true)
	count_posivite=tps[-1]
	count_negative=count-count_posivite

	#计算最大F1
	tns=count_negative-fps
	acc = (tns +tps) / count
	index = np.argmax(acc)
	acc_max=acc[index]
	#返回最大F1值时的参数
	threshold=thresholds[index]
	TP=tps[index]
	FP=fps[index]
	FN=count_posivite-TP
	TN=count_negative-FP
	return TP,FP,FN,TN,threshold,acc_max

def compute_with_max_recall(y_true, y_score, pos_label=None,
						   sample_weight=None):
	"""返回最大F1值时的 TP,FP,FN,TN,threshold,F1_max
	"""
	y_true=np.array(y_true).flatten()
	y_score=np.array(y_score).flatten()
	fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
											 pos_label=pos_label,
											 sample_weight=sample_weight)
	precision = tps / (tps + fps)
	precision[np.isnan(precision)] = 0
	recall = tps /tps[-1]
	count=np.size(y_true)
	count_posivite=tps[-1]
	count_negative=count-count_posivite
	# stop when full recall attained
	# and reverse the outputs so recall is decreasing
	last_ind = tps.searchsorted(tps[-1])
	# sl = slice(last_ind, None, -1)
	# thresholds=thresholds[sl]
	#返回最大F1值时的参数
	threshold=thresholds[last_ind]
	TP=tps[last_ind]
	FP=fps[last_ind]
	FN=count_posivite-TP
	TN=count_negative-FP
	return TP,FP,FN,TN,threshold

if __name__=="__main__":
	#print("我的name{%4d}".format(1000))
	print("wo de {key:010d}".format(key=10000))