import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


# 读取csv文件
def read_csv_(train, test, flag_of_evaluate, submission=None):
    """
    Read_csv_()函数：
            train -> str : 训练数据文件的路径
            test -> str : 测试数据文件的路径
            submission -> str : 测试集的真实y值，评测阶段需输入，预测时不需要，默认为None
    """
    train_df = pd.read_csv(train, index_col=0)
    test_df = pd.read_csv(test, index_col=0)
    if flag_of_evaluate == 1:
        y_test_df = pd.read_csv(submission, index_col=0)
        return train_df, test_df, y_test_df
    return train_df, test_df


# 对数据进行预处理
# 数据平滑处理
def smooth_data(y):
    """
    平滑数据
    y -> str : 训练集中初始的y
    """
    y_train = np.log1p(train_df.pop(y))
    return y_train


# One-Hot编码
def one_hot(num_str=False, ncol_list=None):
    """
    对类别型数据进行One-Hot编码
    num_str -> bool : 是否需要将用数字表示的类别型数据变回字符串类型
    ncol_list -> list : 需要转为字符串类型的列索引组成的列表
    """
    all_df = pd.concat((train_df, test_df), axis=0)  # 暂时合并训练、测试集，便于之后的处理
    if num_str is True:
        all_df = num_to_str(all_df, ncol_list)
    all_dummy_df = cato_to_num(all_df)
    return all_df, all_dummy_df


# 将用数字表示的类别型数据变回字符串类型
def num_to_str(all_df, ncol_list):
    """
    将用数字表示的类别型数据变回字符串类型
    ncol_list -> list : 需要转为字符串类型的列的列名组成的列表
    """
    all_df[ncol_list] = all_df[ncol_list].astype(str)
    return all_df


# 将类别型数据变成数值型数据，通过One-Hot（独热码）编码进行数据变换
def cato_to_num(all_df):
    """
    将类别型数据变成数值型数据，通过One-Hot（独热码）编码进行数据变换
    """
    all_dummy_df = pd.get_dummies(all_df)
    return all_dummy_df


# 处理缺失值
def handle_nan(all_dummy_df):
    """
    处理缺失值
    """
    # 用每列的平均值填补缺失值
    mean_cols = all_dummy_df.mean()
    all_dummy_df = all_dummy_df.fillna(mean_cols)
    return all_dummy_df


# 标准化数据
def normalize(choice, all_df, all_dummy_df):
    """
    标准化数据
    chioce -> int : 1 方法1：(X - X.mean) / X.std
                    2 方法2：preprocessing标准化数据模块
    """
    numeric_cols = all_df.columns[all_df.dtypes != 'object']
    if choice == 1:
        all_dummy_df = normalize1(numeric_cols)
    elif choice == 2:
        all_dummy_df = normalize2(numeric_cols)
    else:
        print("请输入1或2")
    return all_dummy_df


def normalize1(numeric_cols):
    """
    数据标准化方法1：(X - X.mean) / X.std
    """

    numeric_col_mean = all_dummy_df.loc[:, numeric_cols].mean()
    numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
    all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_mean) / numeric_col_std
    return all_dummy_df


def normalize2(numeric_cols):
    """
    数据标准化方法2：preprocessing标准化数据模块
    """
    all_dummy_df.loc[:, numeric_cols] = preprocessing.scale(all_dummy_df.loc[:, numeric_cols])
    return all_dummy_df


# 划分训练/测试集
def divide_sets(all_dummy_df):
    """
    划分训练/测试集
    """
    dummy_train_df = all_dummy_df.loc[train_df.index]
    dummy_test_df = all_dummy_df.loc[test_df.index]
    return dummy_train_df, dummy_test_df


# 选取模型
def choose_model(model_chosen, dummy_train_df, dummy_test_df):
    """
    选取模型.

    model_chosen -> str :选取的模型名称
    """
    x_train = dummy_train_df.values
    x_test = dummy_test_df.values

    modeldict = {
        'ridge': ridge, 'linearregression': linearregression
    }

    model = modeldict.get(model_chosen, unknown_model)(x_train, y_train)
    if model == 0:
        print("不存在此模型")
    else:
        return model


# 未知模型
def unknown_model(x_train, y_train):
    """
    当用户输入未知的模型名称时调用此函数.

    :return: 提示语
    """
    pass
    return 0


# 岭回归
def ridge(x_train, y_train):
    """
    岭回归

    """
    # 用户自定义测试参数的范围
    a, b = map(float, input("请输入想要交叉验证的参数alpha的范围(闭区间[a,b]的a,b值，用空格隔开：").split())
    n = int(input("测试参数的个数："))
    print("请稍等")
    alphas = np.logspace(a, b, n)
    test_scores = []
    for alpha in alphas:
        clf = Ridge(alpha)
        test_score = -cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
        test_scores.append(np.mean(test_score))
    plt.plot(alphas, test_scores)
    plt.title('Alpha vs CV Error')
    plt.show()
    # 用户通过看图选择一个合适的参数best_alpha,如：15
    c = int(input("请根据图像选择一个合适的参数alpha:"))
    model = Ridge(c)
    return model


# 线性回归
def linearregression(x_train, y_train):
    """
    线性回归

    """
    model = LinearRegression()
    return model
    # model.fit(X_train, y_train)
    # pred_y_L = np.expm1(model.predict(X_test))
    # print(pred_y_L)


# 拟合
def fit(model, x_train, y_train):
    """
    拟合
    :return:
    """
    model_fit = model.fit(x_train, y_train)
    return model_fit


# 评测/预测
def pred(model_fit, x_test, flag_of_smooth):
    """
    评测/预测

    """
    if flag_of_smooth == 1:
        pred_y = np.expm1(model_fit.predict(x_test))
    else:
        pred_y = model_fit.predict(x_test)
    return pred_y


# 存储预测结果
def save(pred_y):
    s_pred_y = pd.Series(pred_y)
    d = {'Id': test_df.index, 'SalePrice': s_pred_y}
    result_df = pd.DataFrame(d)
    result_df.to_csv("D:/Machine Learning/kaggle_house_price/result_of_linearregression.csv", encoding='utf-8',index=False)

    # np.savetxt('D:/Machine Learning/kaggle_house_price/prediction.txt', pred_y, delimiter='\n')


# # 查看得分
# def show_score() -> None:
#     """
#     显示模型拟合的效果得分
#     :return:
#     """
#     print(model.score(x_test, y_test))


if __name__ == '__main__':

    # 读取csv
    print("------读取数据------")
    flag_of_evaluate = int(input("是否是评测阶段（0：不是，1：是）："))
    if flag_of_evaluate == 0:
        train_df, test_df = read_csv_('D:/Machine Learning/kaggle_house_price/train.csv',
                                      'D:/Machine Learning/kaggle_house_price/test.csv',
                                      flag_of_evaluate)
    else:
        train_df, test_df, y_test_df= read_csv_('D:/Machine Learning/kaggle_house_price/train.csv',
                                                'D:/Machine Learning/kaggle_house_price/test.csv',
                                                flag_of_evaluate,
                                                'D:/Machine Learning/kaggle_house_price/sample_submission.csv')

    # 数据预处理
    print("------数据预处理------")
    # 数据平滑处理
    flag_of_smooth = int(input("是否需要对数据进行平滑处理（0：不需要，1：需要）："))
    if flag_of_smooth == 1:
        y_train = smooth_data('SalePrice')
    else:
        y_train = train_df.pop('SalePrice')
    # One-Hot编码处理
    flag_of_onehot = int(input("是否需要对数据进行One-Hot编码处理（0：不需要，1：需要）："))
    if flag_of_onehot == 1:
        all_df, all_dummy_df = one_hot(num_str=True, ncol_list=['MSSubClass'])
    else:
        all_df = pd.concat((train_df, test_df), axis=0)
    # 处理缺失值
    flag_of_nan = int(input("是否需要处理数据中的缺失值（0：不需要，1：需要）："))
    if flag_of_nan == 1:
        if flag_of_onehot == 1:
            all_dummy_df = handle_nan(all_dummy_df)
        else:
            all_df = handle_nan(all_df)
    # 数据标准化
    flag_of_norm = int(input("是否需要标准化数据（0：不需要，1：需要）："))
    if flag_of_norm == 1:
        if flag_of_onehot == 1:
            all_dummy_df = normalize(1, all_df, all_dummy_df)
        else:
            all_df = normalize(1, all_df, all_df)

    # 划分训练/测试集
    print("------正在划分训练/测试集------")
    if flag_of_onehot == 1:
        dummy_train_df, dummy_test_df = divide_sets(all_dummy_df)
        x_train = dummy_train_df.values
        x_test = dummy_test_df.values
    else:
        all_train_df, all_test_df = divide_sets(all_df)
        x_train = all_train_df.values
        x_test = all_test_df.values

    # 选取模型
    print("------选取模型------")
    model_chosen = input("请输入选取的模型的名称（小写）：")
    if flag_of_onehot == 1:
        model = choose_model(model_chosen, dummy_train_df, dummy_test_df)
    else:
        model = choose_model(model_chosen, all_train_df, all_test_df)

    # 拟合
    print("------拟合------")
    model_fit = fit(model, x_train, y_train)

    # 预测
    print("------预测------")
    print("所选取模型的预测结果为：")
    pred_y = pred(model_fit, x_test, flag_of_smooth)
    print(pred_y)
    print(type(pred_y))

    # 存储预测结果
    flag_of_save = int(input("是否需要将预测结果存入文件（0：不需要，1：需要）："))
    if flag_of_save == 1:
        save(pred_y)
