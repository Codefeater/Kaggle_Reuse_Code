import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,NMF
from sklearn.cluster import KMeans

'''
feat_present:生成文章，以及文章段落
featf : 对文章进行tfidf
concat_sparse_svd：对SVD降低维数之后，加在train_test中
'''
def feat_present(train_test,begin ,end):
    aux1 = []
    aux2 = []
    for i in train_test.groupby(begin)[end]:
        aux1.append(i[0])
        aux2.append(','.join(i[1].values))
    res = pd.DataFrame({
        begin:aux1,
        begin+"__"+end:aux2
    })
    train_test = train_test.merge(res,on = begin,how = "left")
    return train_test

def Featf(train_test,feat_name):
    tfidf_1 = TfidfVectorizer(stop_words='english')
    csr_1 = tfidf_1.fit_transform(train_test[feat_name])
    train_test.drop(feat_name,axis = 1, inplace = True)
    return train_test,csr_1


def concat_sparse_svd(train_test, n_components,begin,end):
	train_test = feat_present(train_test,begin,end)
	train_test,csr_1 = Featf(train_test,begin + "__" + end)
	csr_1_svd = pd.DataFrame(TruncatedSVD(n_components=n_components).fit_transform(csr_1))
	csr_1_svd.columns = [begin + "_to_" + end + str("_") + str(i) for i in range(n_components) ]
	train_test = pd.concat([train_test, csr_1_svd], axis = 1)
	return train_test
