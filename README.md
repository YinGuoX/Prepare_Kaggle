# Python数据分析与机器学习
* 主要参考于唐宇迪的Python数据分析与机器学习实战(B站有视频)
## 1.Numpy基础
主要是用来矩阵计算，更多请即用即查！！！


与Pandas不同：
* axis=1:一般是处理行数据
* axis=0:一般是处理列数据

* 直接查看对应函数的帮助文档
	*  print(help(numpy.genfromtxt))
* 从txt文件中读取数据
	*  world_alcohol = numpy.genfromtxt("world_alcohol.txt", delimiter=",",dtype=str)
* ndarry的索引，切片，改变值==list的一样
* 创建ndarray数据
	*  np.ones( (2,3,4), dtype=np.int32 )
	*  np.zeros ((3,4)) 
	*  a = np.arange(15).reshape(3, 5)
	*  随机产生：np.random.random((2,3))
	*  等距定量产生：np.linspace( 0, 2*pi, 100 )
* 查看ndarray数据的形状
	* matrix.shape
* 改变ndarray数据的形状
	* 法一：matrix.reshape(size)
	* 法二：matrix.shape=(size)
	* 法三：在原本数据基础上进行扩展
		* b = np.tile(a, (3,4,6)) 
* 展平ndarray数据
	* a.ravel() 
* 查看ndarray数据的元素个数
	* matrix.size	 
* 查看ndarray数据的元素类型
	* matrix.dtype  
* ndarray数据的判断=>返回bool类型的ndarray(可以进行与或非)=>可以用来索引搜索出为True位置上的值
	* 判断是否存在nan值： numpy.isnan(world_alcohol[:,4])
* 转换ndarray数据的元素类型
	* vertor.astype(float64) 
* ndarray数据求和
	* 全部求和：matrix.sum()
	* 求每一行的和：matrix.sum(axis=1)
	* 求每一列的和：matrix.sum(axis=0)
* ndarray数据求平均值(与sum()类似)
* ndarray数据的数学运算（分清楚是对元素计算和整体数据计算）
	* 对应位置相乘：A*B
	* 正规的矩阵乘法(等价)
		* A.dot(B)
		* np.dot(A, B)
	* np.exp(B)
	* np.sqrt(B)
	* 向下取整：
		* a = np.floor(10*np.random.random((3,4))) 
	* 转置： a.T
	* 找最值索引
		* 每一列最大值的索引：ind = data.argmax(axis=0)
	* 排序
		* 按行排序：   b=np.sort(a,axis=1)
		* 获得排序后的元素索引：
			* j = np.argsort(a)
			* print(a[j])
* ndarray数据拼接
	* 水平拼接：np.hstack((a,b))
	* 垂直拼接：np.vstack((a,b))
	* 数组拼接：numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接
* ndarray数据分割
	* 水平分割：np.hsplit(a,3)
	* 垂直分割：np.vsplit(a,3)
* ndarray数据的复制（三法）
	* 两个就是同一个，无论哪个改变另一个一样变
		* b = a
		* print(b is a)
		* print(id(a))
		* 改变一个的形状和大小，和另一个比较可得结论
	* 两个不是同一个，但是共用元素
		*  c = a.view()
		*  print(c is a)
		*  print(id(c))
		*  改变一个的形状和大小，和另一个比较可得结论
	* 两个不是同一个，也不共用元素
		*  d = a.copy() 
		*  print(d is a)
		*  print(id(d))
		*  改变一个的形状和大小，和另一个比较可得结论

## 2.Pandas基础
主要是用来数据处理，更多请即用即查！！！

与Numpy不同：
* axis=1:一般是处理列数据
* axis=0:一般是处理行数据

**DataFrame数据类型：**

* 从csv文件中读取数据
	
	* food_info = pandas.read_csv("food_info.csv")
* 查看数据列名
	
	* food_info.columns
* 查看数据形状(维度)
	
	* food_info.shape
* 获取某几行数据
	* food_info.loc[0]
	* food_info.loc[3:6]
	* food_info.loc[[2,5,10]]
	* iloc()：通过行号(索引号)来取行数据
* 获取某几列数据
	* food_info["NDB_No"]
	* food_info[["Zinc_(mg)", "Copper_(mg)"]]
	* 获取指定列名数据
		* col_names = food_info.columns.tolist()
		* for c in col_names:
			*  if c.endswith("(g)"):
				*  gram_columns.append(c)
			* food_info[gram_columns]
* 获取具体某个数据
	
	* row_index_83_age = titanic_survival.loc[83,"Age"]
* 数学运算：加减乘除...
	* 与numpy类似，大部分都是对每个元素进行计算or对应位置元素的计算
	* food_info["Iron_(mg)"] / 1000
	* add_100 = food_info["Iron_(mg)"] + 100
	* mult_2 = food_info["Iron_(mg)"]*2
	* food_info["Water_(g)"] * food_info["Energ_Kcal"]
	* 找最值
		* food_info["Energ_Kcal"].max()
	* 排序
		* 根据某一列排序，并且是否改变原来数据
			* food_info.sort_values("Sodium_(mg)", inplace=True)
			* 排序顺序：添加ascending=False 
	* 平均值，可以自动忽略nan值
		* correct_mean_age = titanic_survival["Age"].mean()
* 数据拼接(添加)
	* 添加一列(维度要相同)
		* food_info[new_column_name] = new_column
* 数据判断(一般处理为nan值，其余判断类似numpy数据)
	* 判断某一列是否为nan,返回对应索引的bool值
		* age_is_null = pd.isnull(titanic_survival["Age"])
		* 获得为nan值的数据
			* age_null_true = age[age_is_null]
		* 获取为nan值的个数
			* age_null_count = len(age_null_true)
	* 获得无缺失值数据
		* good_ages = titanic_survival[columns_name][age_is_null == False]
	* 处理nan对应的行，列数据
		* axis=1只要数据中有nan就把对应的列数据去掉
			* drop_na_columns = titanic_survival.dropna(axis=1)
		* 在subset列中只要有Nan就把对应的行去掉，因为axis=0
			* new_titanic_survival = titanic_survival.dropna(axis=0,subset=["Age", "Sex"])
* 数据透视表：pivot_table()，建议自行百度案例学习
	* 根据index进行分组计算
	* 据values选择我们需要的列=>一般是根据分好组后的列直接计算平均
	* 再进行 aggfunc=np.mean操作,是对values对应的列进行操作
	* 如：计算不同阶层的获救人数的平均数
		* passenger_survival = titanic_survival.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
	* 计算不同阶层的平均年龄
		* passenger_age = titanic_survival.pivot_table(index="Pclass", values="Age")
	* 计算不同登船点的总费用和获救人数
		* port_stats = titanic_survival.pivot_table(index="Embarked", values=["Fare","Survived"], aggfunc=np.sum)
* 重新设置索引
	* fandango_films = fandango.set_index('FILM', drop=False)
* 重新排序索引
	* 如：对数据按某列排序后，重新排序索引
		* new_titanic_survival = titanic_survival.sort_values("Age",ascending=False)
		* titanic_reindexed = new_titanic_survival.reset_index(drop=True)
		* #drop=True:删除原来的索引
* apply()：自定义函数的使用 ，建议自行百度案例学
	* 如：获取第一百行的数据
		* 定义一个获取一百行的函数： hundredth_row(dataframe)
		* 将titanic_survival传入到hundredth_row函数中并调用
		* hundredth_row = titanic_survival.apply(hundredth_row)
	* 如：判断某一列的为nan的个数
		* 定义一个获取某一列为nan的函数：not_null_count(dataframe)
		* column_null_count = titanic_survival.apply(not_null_count)
	* 更多使用请自行脑补


**Series数据类型：**

一般DataFrame数据的某一列就是一个Series数据类型

* 构造Series数据
	* series_film = fandango['FILM']
	* film_names = series_film.values
	* rt_scores = series_rt.values
	* series_custom = Series(rt_scores , index=film_names)
* Series的索引，切片，改变值==list的一样，并且还可以使用str来索引
* 数学运算
	* 加法：需要索引相同
		* np.add(series_custom, series_custom)
	* 排序
		* 按索引排序(法一)
			* original_index = series_custom.index.tolist()
			* sorted_index = sorted(original_index)
			* sorted_by_index = series_custom.reindex(sorted_index)
		* 按索引排序(法二)
			* sc2 = series_custom.sort_index()
		* 按值排序
			* sc3 = series_custom.sort_values() 
* 数据判断
	* criteria_one = series_custom > 50
	* criteria_two = series_custom < 75
	* both_criteria = series_custom[criteria_one & criteria_two]
* apply()也可以用来获取Series数据
	* deviations = float_df.apply(lambda x: np.std(x))

## 3.Matplotlib基础

默认都是绘制折线图，如果想要绘制其他图像，改变传入数据部分的代码即可，其余显示图像操作等均相同！
* 如：
	* 柱状图：
		* 垂直柱状图：ax.bar(bar_positions, bar_heights, 0.5)  
		* 水平柱状图：ax.barh(bar_positions, bar_widths, 0.5)
	* 散点图：
		* ax.scatter(x1, y1)  
	* 直方图：
		* ax.hist(norm_reviews['Fandango_Ratingvalue']) 
		* bins:自动将数据划分为指定个数区域：
			* 如对某些数据进行计数时，太多太散，指定区间计数即可
			* ax.hist(norm_reviews['Fandango_Ratingvalue'],bins=20)
			* 指定x的范围：
				* ax.hist(Series数据, range=(4, 5),bins=20) 
	* 盒图：
		* 由五个数值点组成：最小值(min)，下四分位数(Q1)，中位数(median)，上四分位数(Q3)，最大值(max)。
		* 可以：
			* 直观地识别数据集中的异常值(查看离群点)。
			* 判断数据集的数据离散程度和偏向(观察盒子的长度，上下隔间的形状，以及胡须的长度)。
		* ax.boxplot(norm_reviews['RT_user_norm'])   


* 获得画图区域：
	* plt.plot() or fig, ax = plt.subplots()
	* 传入需要绘图的数据并绘图：
		* plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
	* 获得多个画图区域
		* 获得画布：
			* fig = plt.figure()
			* 定义画图区域大小：fig = plt.figure(figsize=(3, 3))
		* 添加子图并且选定位置：
			* ax1 = fig.add_subplot(3,2,1)
			* ax2 = fig.add_subplot(3,2,2)
		* 向子图添加绘图的数据
			* ax2.plot(np.arange(10)*3, np.arange(10))
	* 在同一个图中画多个数据：只要在同一个子图中多次分别传入数据即可
		* plt.plot(x1,y1, c='red')
		* plt.plot(x2,y2, c='blue')

* 完善图像
	* 添加显示数据的颜色、标签、宽度
		* 在添加数据时指定即可（子图亦是）
			* plt.plot(x2,y2, c='blue',linewidth=10 )   
			* plt.plot(x1,x2, c=colors[i], label=label)
			* 改变颜色也可传入三元组：
				* cb_orange = (255/255, 128/255, 14/255)
				* ax.plot(x1,x2, c=cb_orange, label=label)
	* 显示每条数据(线)的标签：
		* loc:定义标签的位置 
		* plt.legend(loc='best')
	* 改变x轴坐标显示：
		* 将x轴坐标显示旋转45度：plt.xticks(rotation=45)
	* 显示x,y刻度：
		* 如果是子图，则使用：
			* ax.set_yticks(tick_positions)
			* ax.set_yticklabels(num_cols)  
	* 限制x,y刻度范围：
		* 如果是子图，则使用：
			* ax.set_ylim(0, 5)  
	* 添加x,y轴坐标名字标签
		* plt.xlabel('Month')
		* plt.ylabel('Unemployment Rate')
		* 如果是子图，则使用：
			* ax.set_xlabel('Fandango')
			* ax.set_ylabel('Rotten Tomatoes')
	* 添加图像名字标签
		* plt.title('Monthly Unemployment Trends, 1948') 
		* 如果是子图，则使用：
			* ax.set_title('x')  
	* 显示图像上下左右的刻度显示：
		* ax.tick_params(bottom="off", top="off", left="off", right="off")
	* 不显示图像上下左右的方框线：
		* for key,spine in ax.spines.items():
			* spine.set_visible(False)
	* 在图像指定位置显示文本内容：
		* ax.text(x, y, 'Women')  
* 展示图
	* 无论是否有子图ax等，只要plt.show()就可都显示 
	* plt.show()  



## 4.Seaborn基础
提供一个更高级界面，用于绘制引人入胜且内容丰富的统计图形，只是在Matplotlib上进行了更高级的API封装，从而使作图更加容易。更多请即用即查！
* 设置图像主题风格：
	* 默认风格：sns.set()
	* 有刻度线：sns.set_style("whitegrid")
	* 有背景无刻度线：sns.set_style("dark")
	* 有背景无刻度线：sns.set_style("white")
	* 无背景有刻度线：sns.set_style("ticks")
	* 设置不同子图不同风格：
		* with sns.axes_style("darkgrid"):
			* plt.subplot(211)
			* 绘制子图
			* sinplot()
		* plt.subplot(212)
		* sinplot(-1)
* 绘制各类图像：
	* 直方图：
		* sns.distplot(x,kde=False)
		* sns.distplot(x, bins=20, kde=False)  
		* 查看数据分布状况：sns.distplot(x, kde=False, fit=stats.gamma)
	* 散点图+直方图：观测两个变量之间的分布关系
		* sns.jointplot(x="x", y="y", data=df)
		* sns.jointplot(x=x, y=y, kind="hex", color="k")
	* 展示变量两两之间的关系：(最好自行百度案例理解)
		* sns.pairplot(irsi)  
		* or:
			* g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
			* g.map(plt.scatter)
	* 回归关系：
		* sns.regplot(x="total_bill", y="tip", data=tips) 
		* sns.lmplot(x="total_bill", y="tip", data=tips)
	* 分类散点图：
		* sns.stripplot(x="day", y="total_bill", data=tips)
	* 分簇散点图：数据点不重叠的分类散点图
		* sns.swarmplot(x="day", y="total_bill", data=tips)
	* 盒图：
		* sns.boxplot(x="day", y="total_bill", hue="time", data=tips);
	* 小提琴图：
		* sns.violinplot(x="total_bill", y="day", hue="time", data=tips)
	* 条形图：
		* sns.barplot(x="sex", y="survived", hue="class", data=titanic)
	* 点图：类似分类折线图
		* sns.pointplot(x="sex", y="survived", hue="class", data=titanic) 
	* 多层面板分类图：指定不同参数，该函数可以绘制不同类型的图
		* sns.factorplot() 
		* 自行百度参数
	* 在数据集的不同子集绘制同一图的多个实例(最好自行百度案例理解)
		* 如以time数据分别对tip数据绘制直方图：
			* g = sns.FacetGrid(tips, col="time")
			* g.map(plt.hist, "tip")
		* 还可以map绘制出各种散点图、盒图 等
		* or：g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
		* g.map(plt.scatter)
	* 热力图：(最好自行百度案例理解)
		* 一般用来画特征相关系数的图
		* 常用于展示一组变量的相关系数矩阵
		*  heatmap = sns.heatmap(uniform_data)
* 完善图像：
	* 设置颜色：
		* 使用：在绘图实例中传入cmap=pal参数即可 
		* sns.palplot(current_palette)可以显示色板颜色
		* sns.color_palette()：可以传入任何颜色，不传参数则有默认的颜色
			* 分类色板：sns.color_palette()
			* 圆形色板：
				* 按某种方式生成8种颜色： sns.color_palette("hls", 8)
				* 控制颜色和饱和度的hls生成颜色方式：sns.hls_palette(8, l=.7, s=.9)
				* 绘制一对一对的元素，一深一浅：sns.color_palette("Paired",8)
			* 使用xkcd颜色来绘制：
				* plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)  
				* sns.xkcd_palette(colors)
			* 连续(渐变)色板：
				* sns.color_palette("Blues")
				* 翻转渐变：sns.color_palette("BuGn_r")
				* sns.light_palette("green")
				* sns.dark_palette("purple")
			* 线性调色板：色调线性变换
				* sns.color_palette("cubehelix", 8)
				* sns.cubehelix_palette(8, start=.5, rot=-.75)
	* 去除上面和右边的方框：sns.despine()
	* 设置图离轴的距离：sns.despine(offset=100)
	* 指定隐藏轴：sns.despine(left=True)
	* 设置图的样式：
		* sns.set_context("paper") or talk or poster or notebook等
		* sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
* 图像显示：
	* 照着matplotlib的plt.plot()or只要调用了sns应就会自动显示
	* 同一张图绘制不同类型的图：顺序绘制即可
		* sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
		* sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5) 


## 5.机器学习算法
### 5.1线性回归
* 具体数学推导看PPTor百度
* 主要推导思路：
* 该问题可用线性回归模型建模=>获得一个线性方程h(x)
* 因为预测肯定与真实存在误差=>假设每个值的误差都是独立同分布的正态分布
* 从误差的正态分布=>概率=>贝叶斯=>在参数与特征的前提下使h(x)=y的概率最大
* 似然函数=>对数似然=>化简=>目标函数J(theta)=>使目标函数最小即可求解
* 求解目标函数最小：=>都假设满足凸优化
	* 法一：求偏导使为0求解可得
		* 缺陷：特征不一定可逆，x越大计算量越大
	* 法二： 梯度下降：不断找方向下降
		* 缺陷：可能陷入局部最优解
		* 梯度下降三法
			* 批量梯度下降
			* 随机梯度下降
			* 小批量梯度下降 
		* 学习率(步长)的选择：动态调整，不断变小 
* 评估方法： 越接近1模型拟合越好
$$
R^2=1-\frac{\sum_{i=1}^{m}(y_i^{pred}-y_i)^2}{\sum_{i=1}^{m}(y_i-y_i^{avg})^2}
$$

### 5.2逻辑回归
* 具体数学推导看PPTor百度
* 逻辑回归在线性回归的基础上进行推导用来二分类
* 主要推导思路：
* 该问题可用逻辑回归模型建模=>获得一个概率预测函数h(x)=g(theta*x)
* h(x)获得的就是对于的概率值=>进行全域的整合=>似然函数=>对数函数=>求最大=>求最小
* 转变为梯度下降任务求解即可

### 5.3 录取分类模型-逻辑回归
主要构建思路：**分模块构建**

主要思路：

* 数据处理
	* 代码技巧： 
	* 读取数据，并且重新设置列名：
		* pdData = pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
	* 找满足某列条件的全部数据：
		* positive = pdData[pdData['Admitted']==1] 
* 构建逻辑回归模型(分步构建不断包含)
	* 代码技巧：
		* 将DataFrame数据转换为martix数据：orig_data = pdData.values 
		* 打乱数据集：np.random.shuffle(data)
	* sigmoid(z)：映射到概率的函数
	* model(X,theta)：返回概率的函数=>计算h(x)
		* 包含sigmoid() 
	* cost(X,y,theta)：计算误差(损失)：计算对数似然函数的平均损失
	* gradient(X,y,theta)：根据对数似然函数的平均损失计算参数的梯度
		* 包含model()，因为要计算h(x) 
	* descent(data,theta,batchSize,stopType,thresh,alpha)：梯度更新=>三种不同的停止梯度下降的方法
		* 包含gradient(),cost()
	* runExpe(data,theta,batchSize,stopType,thresh,alpha)：整体运行并且打印信息绘制图像
		* 包含descent()  
	* accuracy：计算预测精度
		* 包含predict(X,theta) 
* 实践结果
	* 对比了三种梯度下降的方法：批量梯度下降、随机梯度下降、小批量梯度下降
	* 学习率对梯度下降的影响：过大容易误差波动大
	* 数据标准化对模型有益=>**数据预处理是很有必要的**

### 5.4 信用卡欺诈分类-逻辑回归
数据拿来就已经做好了特征工程，进行建模分类即可，主要是强调**数据不均衡**如何处理的问题

主要思路：

* 数据处理
	* 代码技巧：
	* 查看某列中的不同值并且计算相同值的个数
		* pd.value_counts(data['Class'],sort=True)
	* 数据归一化：因为其他特征都是在一定范围，而另一个特征范围较大，因此考虑对其进行归一化处理
		* data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
* 数据不均衡如何处理？
	* 过采样：生成样本，使样本同样的多
		* SMOTE算法(自行百度算法原理)
			* oversample = SMOTE(random_state=0)
			* os_features,os_labels = oversample.fit_resample(features_train,labels_train)
	* 下采样：抽取样本，使样本同样少
		* 根据样本少的数据随机从样本多的数据中抽取等量的数据
* 交叉验证集(自行百度具体思想)
	* 先数据划分：	 
	* X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)   
	* 再设置交叉验证：
	* fold = KFold(5,shuffle=False)
	* for train_index,test_index in fold.split(x_train_data):...
* 评估函数
	* 采用Recall=TP/(TP+TF) 
	* 为什么用Recall作为模型评估标准而不是accuracy,因为accuracy在样本不均衡情况下不能正确的评估模型
* 建立逻辑回归模型：(具体看代码即可)
	* 采用sklearn构建即可
	* 考虑超参数的选择！ 
		* 如：正则化参数、sigmoid的阈值 
	* lr = LogisticRegression(C=c_param,penalty='l1',solver='liblinear')
	* lr.fit(X,y) 
	* lr.predict(X)
	* recall_score(...)
* 绘制混淆矩阵
	* 查看模型效果
	* 使用下采样数据训练模型
	* 使用原数据训练模型
	* 使用过采样数据训练模型 


### 5.5决策树
根据特征属性划分数据，构建决策树模型
* 组成：
	* 根节点
	* 非叶子节点
	* 叶子节点
* 如何切分特征?(如何选择节点？)
	* 目标：通过一种衡量标准，来计算选择不同的特征后分类的情况，找出最好的当做节点即可，不断迭代！ 
	* 衡量标准：熵
	* 决策节点：
		* 信息增益
* 根据不同的决策演变的不同决策树算法：
	* ID3：信息增益
	* C4.5：信息增益率
	* CART：GINI系数
* 决策树问题：
	* 处理连续值=>离散化
	* 处理过拟合：
		* 剪枝策略
			* 预剪枝  
			* 后剪枝
* 调参 

### 5.6 房价预测-决策树
* 数据处理：使用sklearn内置数据
* 建立决策树模型：使用sklearn内置库(官网API)
	* 可视化决策树 
* 建立随机森林模型：
	* 网格搜索法选择超参数 
	* 查看哪些特征比较重要


### 5.7 集成学习(Ensemble learning)
* 目的：让机器学习效果更好(单个不行就一起上)

* 主要分为三大类：
* Bagging(bootstrap aggregation)：并行思想=>训练多个分类器取平均
	* 随机森林：
		* 随机：采样数据随机、特征选择随机
		* 森林：多个决策树并行放在一起
		* 优势：可处理高纬度数据、可以给出哪些特征比较重要、并行速度快、易可视化分析 
* Boosting：串行思想=>从弱学习器开始加权，通过加权来进行训练
	* AdaBoost：
		* 思想：根据前一次的分类效果调整数据权重
		* 结果：每个分类器根据自身的准确性来确定各自的权重
	* Xgboost 

* Stacking：直接堆叠模型，将第一阶段一堆模型的结果作为下一阶段模型的输入
	* 为了刷分，不择手段 

### 5.8 泰坦尼克号获救预测-线性回归、逻辑回归、随机森林、集成学习
* 代码技巧：
	* 查看数据基本信息：titanic.describe()
	* 缺失值填充：titanic["Age"] = titanic["Age"].fillna(titanic['Age'].median())
	* 查看某一列有哪些值：titanic['Sex'].unique()
	* 查看某一列有哪些值且计算重复数量：titanic['Embarked'].value_counts()
	* 将类别数值化：titanic.loc[titanic['Sex']=="male","Sex"]=0
* 建立模型：sklearn的使用
	* 注意：
		* 实例化分类器需要加()
		* KFold()使用方法已经改变，最好看官方API
	* LinearRegression() 
	* LogisticRegression()
	* RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=2,min_samples_leaf=1)
* 测试构造新特征：
	* 姓名区别、长度、家庭大小等
	* titanic['FamilySize']=titanic['SibSp']+titanic['Parch'] 
	* titanic['NameLength']=titanic['Name'].apply(lambda x:len(x))
	* 姓名数值化：Mr:1,Miss:2等
* 查看属性主要程度，使用SelectKBest：
	* selector = SelectKBest(f_classif, k=5)
	* selector.fit(titanic[predictors], titanic["Survived"])
	* scores = -np.log10(selector.pvalues_)
	* 绘图...
	* 根据特征主要程度重新选择特征进行建模分类
* 使用集成学习方法： from sklearn.ensemble import GradientBoostingClassifier
	* 最好看官方API 
### 5.9 贝叶斯算法
重点是贝叶斯公式的使用和变体(自行百度即可)
* 相关的衍生：
* 极大似然估计
* 朴素贝叶斯分类器
* 半朴素贝叶斯分类器
* 贝叶斯网
* EM算法
### 5.10 拼写检查器-贝叶斯算法
感觉不能叫做使用了贝叶斯算法的拼写检查器，只是运用的是贝叶斯的思想而已！
* 主要思想：
* 定义一种单词之间的衡量标准：编辑距离
* 根据编辑距离来判定是否是正确的单词
* 如果错误则选择编辑距离最短的那个在语料库中有的单词即可

### 5.11 文本分析(未完待续)
* 文本预处理：文本清洗
	* 清洗停用词
		* 停用词：语料库中大量出现，但是没有用的词
* 关键词提取：Tf-idf
	* Tf(Term Frequency)(词频)： 某个词在文章中出现的次数/文章的总词数
	* idf(Inverse Ducument Frequency)(逆文档频率)：如果某个词比较少见，但是在这篇文章中多见，则该词很有可能就是关键词
		* log(语料库的文档总数/包含该词的文档数)
	* Tf-idf = Tf * idf
* 相似度： 判断两段文本的相似度一般是将文本转换为向量，再使用某种度量方式来进行判断
	* 余弦相似度度量：cos...

### 5.12 新闻主题分类
主要思路：文本分词、文本清洗停用词、获取词向量(词频向量、Tfidf向量)、朴素贝叶斯分类
* 代码技巧：
	* jieba库的使用：注意数据的输入格式
		* jieba.lcut()：文本分词作用
		* jieba.analyse.extract_rags：提取主题词作用
	* gensim库的使用：看官网API吧，我也不会！！！
	* sklearn的使用：看官网API吧，我也不会！！！
		* 获取词频向量：CountVectorizer
		* 获取Tfidf向量： TfidfVectorizer
		* 朴素贝叶斯分类器的使用：MultinomialNB()

### 5.13 支持向量机
* 解决问题：
	* 什么样的决策边界才是最好的?=>离边界点(雷区)最远的==Large Margin
	* 特征数据本身难分怎么办?=>核函数(常用高斯核函数)=>升维=>先映射，再计算=>先计算，再映射(计算复杂度--)
* 主要思路：
	* 找出最好的决策边界=>Large Margin=>如何判断是否是Large Margin=>样本点到假设平面(决策方程)的距离=>distance(x,b,w)
	* 假设约束：样本类别为+1、-1=> y*y(x)>0，决策方程为：y(x)=wx+b
	* 明确总目标：找到一个w,b使得离决策方程最近的点距离最大
	* 根据假设约束改进distance(x,b,w)公式=>通过放缩变化优化目标(y*y(x)>1)=>获取目标函数+约束条件
	* 求解目标极大值转换为求解极小值 
	* 求解思路：带约束的拉格朗日乘子法=>获取目标函数L(w,b,a)
	* 根据对偶问题(KKT)来改进求解过程：对w,b求偏导获等式=>代入L(w,b,a)再对a求导=>代入样本数据(X,Y)=>根据约束条件可求得a=>进而再求的w,b
	* 至此目标求解完毕，得出决策方程
	* 支持向量的由来？
		* 在 L(w,b,a)对a求导获取等式并且代入数据求解a时：
			* a_i = 0的样本点都不会影响到w,b的值=>不是边界的样本点
			* a_i > 0的样本点才会 影响到w,b的值=>是边界上的样本点(是位于最大间隔的点)=>是一个支持向量！！
		* 所以说：训练完成之后，大部分训练样本都没有起到作用，最终模型仅与支持向量有关 

* 以上支持向量机的问题：
	* 如果训练样本数据中存在噪音点，并且噪音点被作为支持向量=>极大影响决策边界！ 
	* 解决办法=>soft-margin！
		* 思路： 引入松弛因子：y*y(x)>1-(松弛因子)
			* 获得新的目标函数 => min 1/2(w)^2 +C*sum(松弛因子)
			* 当C很大时=》意味着松弛因子较小=》分类严格不能有错
			* 当C很小时=》意味着松弛因子较大=》分类宽松可以有错
			* C的选择很重要
