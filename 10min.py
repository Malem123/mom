Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:04:45) [MSC v.1900 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> s = pd.Series([1,3,5,np.nan,6,8])
>>> s
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
>>> dates = pd.date_range('20130101', periods=6)
>>> dates
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
>>> df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
>>> df
                   A         B         C         D
2013-01-01 -0.007133  1.469622 -2.091648  0.223843
2013-01-02 -0.229974  0.729194  1.353872 -0.365873
2013-01-03  0.965405  2.920553  0.521884  1.438479
2013-01-04 -0.464982  0.235445 -0.626671  0.267520
2013-01-05  0.249538  1.875167  0.659052 -0.276356
2013-01-06 -0.337401  1.724628 -0.764658  0.046627
>>> df2 = pd.DataFrame({ 'A' : 1.,
... 'B' : pd.Timestamp('20130102'),
... 'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
... 'D' : np.array([3] * 4,dtype='int32'),
... 'E' : pd.Categorical(["test","train","test","train"]),
... 'F' : 'foo' })
>>> df2
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
>>> df2.dtypes
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
>>> df2.<TAB>
  File "<stdin>", line 1
    df2.<TAB>
        ^
SyntaxError: invalid syntax
>>> df.head()
                   A         B         C         D
2013-01-01 -0.007133  1.469622 -2.091648  0.223843
2013-01-02 -0.229974  0.729194  1.353872 -0.365873
2013-01-03  0.965405  2.920553  0.521884  1.438479
2013-01-04 -0.464982  0.235445 -0.626671  0.267520
2013-01-05  0.249538  1.875167  0.659052 -0.276356
>>> df.tail(3)
                   A         B         C         D
2013-01-04 -0.464982  0.235445 -0.626671  0.267520
2013-01-05  0.249538  1.875167  0.659052 -0.276356
2013-01-06 -0.337401  1.724628 -0.764658  0.046627
>>> df.index
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
>>> df.columns
Index(['A', 'B', 'C', 'D'], dtype='object')
>>> df.values
array([[-0.00713328,  1.46962191, -2.0916477 ,  0.22384309],
       [-0.22997391,  0.72919387,  1.35387239, -0.36587289],
       [ 0.96540539,  2.92055309,  0.52188404,  1.43847903],
       [-0.46498242,  0.23544478, -0.62667063,  0.26751992],
       [ 0.24953779,  1.87516725,  0.6590517 , -0.27635601],
       [-0.33740086,  1.72462774, -0.76465784,  0.04662661]])
>>> df.describe()
              A         B         C         D
count  6.000000  6.000000  6.000000  6.000000
mean   0.029242  1.492435 -0.158028  0.222373
std    0.523857  0.938403  1.244368  0.649092
min   -0.464982  0.235445 -2.091648 -0.365873
25%   -0.310544  0.914301 -0.730161 -0.195610
50%   -0.118554  1.597125 -0.052393  0.135235
75%    0.185370  1.837532  0.624760  0.256601
max    0.965405  2.920553  1.353872  1.438479
>>> df.T
   2013-01-01  2013-01-02  2013-01-03  2013-01-04  2013-01-05  2013-01-06
A   -0.007133   -0.229974    0.965405   -0.464982    0.249538   -0.337401
B    1.469622    0.729194    2.920553    0.235445    1.875167    1.724628
C   -2.091648    1.353872    0.521884   -0.626671    0.659052   -0.764658
D    0.223843   -0.365873    1.438479    0.267520   -0.276356    0.046627
>>> df.sort_index(axis=1, ascending=False)
                   D         C         B         A
2013-01-01  0.223843 -2.091648  1.469622 -0.007133
2013-01-02 -0.365873  1.353872  0.729194 -0.229974
2013-01-03  1.438479  0.521884  2.920553  0.965405
2013-01-04  0.267520 -0.626671  0.235445 -0.464982
2013-01-05 -0.276356  0.659052  1.875167  0.249538
2013-01-06  0.046627 -0.764658  1.724628 -0.337401
>>> df.sort_values(by='B')
                   A         B         C         D
2013-01-04 -0.464982  0.235445 -0.626671  0.267520
2013-01-02 -0.229974  0.729194  1.353872 -0.365873
2013-01-01 -0.007133  1.469622 -2.091648  0.223843
2013-01-06 -0.337401  1.724628 -0.764658  0.046627
2013-01-05  0.249538  1.875167  0.659052 -0.276356
2013-01-03  0.965405  2.920553  0.521884  1.438479
>>> df['A']
2013-01-01   -0.007133
2013-01-02   -0.229974
2013-01-03    0.965405
2013-01-04   -0.464982
2013-01-05    0.249538
2013-01-06   -0.337401
Freq: D, Name: A, dtype: float64
>>> df[0:3]
                   A         B         C         D
2013-01-01 -0.007133  1.469622 -2.091648  0.223843
2013-01-02 -0.229974  0.729194  1.353872 -0.365873
2013-01-03  0.965405  2.920553  0.521884  1.438479
>>>  df['20130102':'20130104']
  File "<stdin>", line 1
    df['20130102':'20130104']
    ^
IndentationError: unexpected indent
>>> df.loc[dates[0]]
A   -0.007133
B    1.469622
C   -2.091648
D    0.223843
Name: 2013-01-01 00:00:00, dtype: float64
>>> df.loc[:,['A','B']]
                   A         B
2013-01-01 -0.007133  1.469622
2013-01-02 -0.229974  0.729194
2013-01-03  0.965405  2.920553
2013-01-04 -0.464982  0.235445
2013-01-05  0.249538  1.875167
2013-01-06 -0.337401  1.724628
>>> df.loc['20130102':'20130104',['A','B']]
                   A         B
2013-01-02 -0.229974  0.729194
2013-01-03  0.965405  2.920553
2013-01-04 -0.464982  0.235445
>>>  df.loc['20130102',['A','B']]
  File "<stdin>", line 1
    df.loc['20130102',['A','B']]
    ^
IndentationError: unexpected indent
>>> df.loc['20130102',['A','B']]
A   -0.229974
B    0.729194
Name: 2013-01-02 00:00:00, dtype: float64
>>> df.loc[dates[0],'A']
-0.007133281791874781
>>> df.at[dates[0],'A']
-0.007133281791874781
>>> df.iloc[3]
A   -0.464982
B    0.235445
C   -0.626671
D    0.267520
Name: 2013-01-04 00:00:00, dtype: float64
>>> df.iloc[[1,2,4],[0,2]]
                   A         C
2013-01-02 -0.229974  1.353872
2013-01-03  0.965405  0.521884
2013-01-05  0.249538  0.659052
>>> df.iloc[1:3,:]
                   A         B         C         D
2013-01-02 -0.229974  0.729194  1.353872 -0.365873
2013-01-03  0.965405  2.920553  0.521884  1.438479
>>> df.iloc[:,1:3]
                   B         C
2013-01-01  1.469622 -2.091648
2013-01-02  0.729194  1.353872
2013-01-03  2.920553  0.521884
2013-01-04  0.235445 -0.626671
2013-01-05  1.875167  0.659052
2013-01-06  1.724628 -0.764658
>>> df.iloc[1,1]
0.7291938717350267
>>> df.iat[1,1]
0.7291938717350267
>>> df[df.A > 0]
                   A         B         C         D
2013-01-03  0.965405  2.920553  0.521884  1.438479
2013-01-05  0.249538  1.875167  0.659052 -0.276356
>>> df[df > 0]
                   A         B         C         D
2013-01-01       NaN  1.469622       NaN  0.223843
2013-01-02       NaN  0.729194  1.353872       NaN
2013-01-03  0.965405  2.920553  0.521884  1.438479
2013-01-04       NaN  0.235445       NaN  0.267520
2013-01-05  0.249538  1.875167  0.659052       NaN
2013-01-06       NaN  1.724628       NaN  0.046627
>>> df2 = df.copy()
>>> df2['E'] = ['one', 'one','two','three','four','three']
>>> df2
                   A         B         C         D      E
2013-01-01 -0.007133  1.469622 -2.091648  0.223843    one
2013-01-02 -0.229974  0.729194  1.353872 -0.365873    one
2013-01-03  0.965405  2.920553  0.521884  1.438479    two
2013-01-04 -0.464982  0.235445 -0.626671  0.267520  three
2013-01-05  0.249538  1.875167  0.659052 -0.276356   four
2013-01-06 -0.337401  1.724628 -0.764658  0.046627  three
>>> df2[df2['E'].isin(['two','four'])]
                   A         B         C         D     E
2013-01-03  0.965405  2.920553  0.521884  1.438479   two
2013-01-05  0.249538  1.875167  0.659052 -0.276356  four
>>> s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
>>> s1
2013-01-02    1
2013-01-03    2
2013-01-04    3
2013-01-05    4
2013-01-06    5
2013-01-07    6
Freq: D, dtype: int64
>>> df['F'] = s1
>>> df.at[dates[0],'A'] = 0
>>> df.iat[0,1] = 0
>>> df.loc[:,'D'] = np.array([5] * len(df))
>>> df
                   A         B         C  D    F
2013-01-01  0.000000  0.000000 -2.091648  5  NaN
2013-01-02 -0.229974  0.729194  1.353872  5  1.0
2013-01-03  0.965405  2.920553  0.521884  5  2.0
2013-01-04 -0.464982  0.235445 -0.626671  5  3.0
2013-01-05  0.249538  1.875167  0.659052  5  4.0
2013-01-06 -0.337401  1.724628 -0.764658  5  5.0
>>> df2 = df.copy()
>>> df2[df2 > 0] = -df2
>>> df2
                   A         B         C  D    F
2013-01-01  0.000000  0.000000 -2.091648 -5  NaN
2013-01-02 -0.229974 -0.729194 -1.353872 -5 -1.0
2013-01-03 -0.965405 -2.920553 -0.521884 -5 -2.0
2013-01-04 -0.464982 -0.235445 -0.626671 -5 -3.0
2013-01-05 -0.249538 -1.875167 -0.659052 -5 -4.0
2013-01-06 -0.337401 -1.724628 -0.764658 -5 -5.0
>>> df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
>>> df1.loc[dates[0]:dates[1],'E'] = 1
>>> df1
                   A         B         C  D    F    E
2013-01-01  0.000000  0.000000 -2.091648  5  NaN  1.0
2013-01-02 -0.229974  0.729194  1.353872  5  1.0  1.0
2013-01-03  0.965405  2.920553  0.521884  5  2.0  NaN
2013-01-04 -0.464982  0.235445 -0.626671  5  3.0  NaN
>>> df1.dropna(how='any')
                   A         B         C  D    F    E
2013-01-02 -0.229974  0.729194  1.353872  5  1.0  1.0
>>> df1.fillna(value=5)
                   A         B         C  D    F    E
2013-01-01  0.000000  0.000000 -2.091648  5  5.0  1.0
2013-01-02 -0.229974  0.729194  1.353872  5  1.0  1.0
2013-01-03  0.965405  2.920553  0.521884  5  2.0  5.0
2013-01-04 -0.464982  0.235445 -0.626671  5  3.0  5.0
>>> pd.isna(df1)
                A      B      C      D      F      E
2013-01-01  False  False  False  False   True  False
2013-01-02  False  False  False  False  False  False
2013-01-03  False  False  False  False  False   True
2013-01-04  False  False  False  False  False   True
>>> df.mean()
A    0.030431
B    1.247498
C   -0.158028
D    5.000000
F    3.000000
dtype: float64
>>> df.mean(1)
2013-01-01    0.727088
2013-01-02    1.570618
2013-01-03    2.281569
2013-01-04    1.428758
2013-01-05    2.356751
2013-01-06    2.124514
Freq: D, dtype: float64
>>> s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
>>> s
2013-01-01    NaN
2013-01-02    NaN
2013-01-03    1.0
2013-01-04    3.0
2013-01-05    5.0
2013-01-06    NaN
Freq: D, dtype: float64
>>> df.sub(s, axis='index')
                   A         B         C    D    F
2013-01-01       NaN       NaN       NaN  NaN  NaN
2013-01-02       NaN       NaN       NaN  NaN  NaN
2013-01-03 -0.034595  1.920553 -0.478116  4.0  1.0
2013-01-04 -3.464982 -2.764555 -3.626671  2.0  0.0
2013-01-05 -4.750462 -3.124833 -4.340948  0.0 -1.0
2013-01-06       NaN       NaN       NaN  NaN  NaN
>>> df.apply(np.cumsum)
                   A         B         C   D     F
2013-01-01  0.000000  0.000000 -2.091648   5   NaN
2013-01-02 -0.229974  0.729194 -0.737775  10   1.0
2013-01-03  0.735431  3.649747 -0.215891  15   3.0
2013-01-04  0.270449  3.885192 -0.842562  20   6.0
2013-01-05  0.519987  5.760359 -0.183510  25  10.0
2013-01-06  0.182586  7.484987 -0.948168  30  15.0
>>> df.apply(lambda x: x.max() - x.min())
A    1.430388
B    2.920553
C    3.445520
D    0.000000
F    4.000000
dtype: float64
>>> s = pd.Series(np.random.randint(0, 7, size=10))
>>> s
0    6
1    6
2    4
3    2
4    2
5    6
6    1
7    4
8    0
9    0
dtype: int32
>>> s.value_counts()
6    3
4    2
2    2
0    2
1    1
dtype: int64
>>> s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
>>> s
0       A
1       B
2       C
3    Aaba
4    Baca
5     NaN
6    CABA
7     dog
8     cat
dtype: object
>>> s.str.lower()
0       a
1       b
2       c
3    aaba
4    baca
5     NaN
6    caba
7     dog
8     cat
dtype: object
>>> df = pd.DataFrame(np.random.randn(10, 4))
>>> df
          0         1         2         3
0 -0.056902  0.116858 -0.225515 -0.587097
1 -0.640436 -0.841475 -0.466758 -0.263995
2  0.593777  2.535072 -0.829729  1.627607
3  0.531762 -0.538640  0.454864  2.060646
4  2.030739 -1.277611  1.319026  0.260466
5  0.458551  1.401763  1.770689  0.614890
6 -0.084648  0.287640 -0.992317  0.296500
7  0.491457  0.500490  0.766279  0.601167
8 -0.015046  0.169736  0.797123 -0.115372
9  0.595506 -0.668822  0.535436 -0.695385
>>> pieces = [df[:3], df[3:7], df[7:]]
>>> pd.concat(pieces)
          0         1         2         3
0 -0.056902  0.116858 -0.225515 -0.587097
1 -0.640436 -0.841475 -0.466758 -0.263995
2  0.593777  2.535072 -0.829729  1.627607
3  0.531762 -0.538640  0.454864  2.060646
4  2.030739 -1.277611  1.319026  0.260466
5  0.458551  1.401763  1.770689  0.614890
6 -0.084648  0.287640 -0.992317  0.296500
7  0.491457  0.500490  0.766279  0.601167
8 -0.015046  0.169736  0.797123 -0.115372
9  0.595506 -0.668822  0.535436 -0.695385
>>> left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
>>> right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
>>> left
   key  lval
0  foo     1
1  foo     2
>>> right
   key  rval
0  foo     4
1  foo     5
>>> pd.merge(left, right, on='key')
   key  lval  rval
0  foo     1     4
1  foo     1     5
2  foo     2     4
3  foo     2     5
>>> df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
>>> ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
>>> ts = ts.cumsum()
>>> ts.plot()
<matplotlib.axes._subplots.AxesSubplot object at 0x032037D0>
>>>