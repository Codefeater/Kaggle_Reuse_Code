# Kaggle_Repeat_Code
复用自己的代码


## Stacking

DM 是调用文件 生成meta_features
basic_parameters 是xgb的参数，方便修改和调用
single_xgb 是xgb.cv的使用，最后的单模型
stack_model是主要的文件，针对不同模型写的stacking 包含回归
submission 是生成最后的提交文件

## 特征复用文件
feat_to_feat是hash特征的表征


## 其余
其余文件没有开源，主要是写的乱，不过根据名称，可以猜到这个单文件做什么的
