# -*- coding: utf-8 -*-
"""
Created on Fri Dec 05 14:46:17 2014

@author: skku
"""

# -*- coding: utf-8 -*- 

# pandas 이용한 데이터분석 시험에 나옴

 

#데이터 값을 모르는 것들은 평균값을 집어 넣음

 

#기본 operate들을 응용해서 데이터 포멧팅하는게 시험

 

#시험에 del 명령어!!, transpose. 컴퓨터가 인식하기엔 transpose시킨게 더 쉽다!!

 

from pandas import Series, DataFrame

 

import pandas as pd

 

import numpy as np

 

lst = [ 4, -7, 5, 3]

 

print lst

 

obj = Series([4, -7, 5, 3])

 

#obj = Series(lst)

 

print obj

 

print obj.values

 

print obj.index

 

obj2 = Series(lst, index=['d','b','a','c'])

 

print obj2

 

obj22= Series(lst, index=[2010,2011,2012,2013])

 

print obj22

 

print obj2['a']

 

obj2['d'] = 6 #데이터 수정

 

print obj2

 

print obj2[['a','c','d']] #순서 섞기

 

obj222=(obj2[obj2>0])

 

print obj222

 

print ('b' in obj2) # 들어있는지 확인

 

sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000} #딕셔너리형태

 

obj3 = Series(sdata)

 

print obj3

 

states = ['Califonia', 'Ohio', 'Oregon', 'Texas']

 

obj4 = Series(sdata, index=states)

 

print obj4

 

print pd.isnull(obj4) #데이터 넣을 때 숫자를 빼먹은 경우 확인때 사용. 

 

print pd.notnull(obj4)

 

print obj3 + obj4

 

obj4.name = 'Population'

 

obj4.index.name = 'State'

 

print obj4

 

data = {'state':['Ohio','Ohio','Ohio','Nevada','Nebada'],'year':[2000,2001,2002,2001,2002], 'pop':[1.5, 1.7, 3.6, 2.4, 2.9]} #알파벳 순으로

 

frame = DataFrame(data)

 

print frame

 

frame = DataFrame(data, columns=['year','state','pop'])

 

print frame

 

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one','two','three','four','five']) #column 추가 - null값으로 들어감

 

print frame2

 

print frame2['state'] #딕셔너리 형태로 접근

 

print frame2.year #attribute 형태로 접근

 

print frame2.ix['three'] #row 단위로 접근하려면 꼭 ix를 써줘야 함!

 

frame2['debt'] = 16.5

 

print frame2

 

frame2['debt'] = np.arange(5)

 

print frame2

 

val = Series([-1.2, -1.5, -1.7], index = ['two','four','five'])

 

frame2['debt'] = val

 

print frame2

 

frame2['eastern'] = (frame2.state == 'Ohio')

 

print frame2

 

del frame2['eastern']

 

print frame2

 

print frame2.T

 

obj6 = Series(range(3),

index=['a','b','c'])

index = obj6.index

print index

print index[1:]

#index[1] = 'd' 이런식으로는 수정 불가능

 

index=pd.Index(np.arange(3))

obj7 = Series([1.5,-2.5,0.], index=index)

print obj7

print obj7.index is index

 

obj = Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])

print obj

obj2 = obj.reindex(['a','b','c','d','e']) #e 는 새로추가됨 nan값으로 나옴

 

print obj2

 

obj3 = obj.reindex(['a','b','c','d','e'], fill_value=0) # nan 대신 0이들어감

print obj3

obj4 = Series(['blue','purple','yellow'],

index=[0,2,4])

obj5 = obj4.reindex([range(6),], method='ffill') # ffill 앞에것으로 채워라

print obj5

 

obj6= obj4.reindex(range(6), method='bfill') #bfill 뒤에걸로 채워라

print obj6

 

obj= Series(np.arange(5.), index=['a','b','c','d','e'])

new_obj = obj.drop('c')

#column 지울때는 del. row 지울때는 drop

print new_obj

 

##################################################################

 

#from sklearn import datasets

#iris = datasets.load_iris()

#digits = datasets.load_digits()

 

#print iris

#print digits

 

from sklearn import svm

from sklearn import datasets

clf = svm.SVC()

iris = datasets.load_iris()

X, y = iris.data, iris.target

clf.fit(X, y) #이게 학습시키는 것

clf.predict(X) #학습한걸 바탕으로 예측!

#shpae을 하면 각 차원별로 (row, column) 

 

#a=X[:np.rint(len(iris.data)*0.6)]

#b=y[:np.rint(len(iris.target)*0.6)]

 

s=np.rint(iris.data.shape[0]*0.6)

clf.fit(X[:s],y[:s])

clf.predict(X[s:])

 

#clf.fit(a,b)

#clf.predict(a)

 

 

#enumerate를 쓰면 i에는 index가, ix에는 index에 해당하는 데이터가 들어감)

sum0=0

sum1=0

sum2=0

 

for i, ix in enumerate(y):

if ix == 0:

sum0 = sum0+1

pass

elif ix ==1:

sum1= sum1+1

pass

else:

sum2= sum2+1

pass

c0 = np.rint(sum0*0.6)

c1= np.rint(sum1*0.6)

c2= np.rint(sum2*0.6)

 

arrx0 = X[:c0]

arry0 = y[:c0]

arrx1 = X[sum0:sum0+c1]

arry1 = y[sum0:sum0+c1]

arrx2 = X[sum0+sum1:sum0+sum1+c2]

arry2 = y[sum0+sum1:sum0+sum1+c2]

 

#a= np.array([[1,2],[3,4]])

#b=np.array([[5,6]])

#np.concatenate((a,b), axis = 0) # axis=0 row 단위로 합한다.]

 

np.concatenate((arrx0,arrx1),axis=0)

