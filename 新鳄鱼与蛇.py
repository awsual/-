import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为支持中文的字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------- 1. 准备数据 ----------
raw_data = {
    "鳄鱼": {
        "扬子鳄": {"身长": 2.5, "体重": 45},
        "直布罗陀鳄鱼": {"身长": 5.0, "体重": 225},
        "森林鳄": {"身长": 6.0, "体重": 420},
        "马来鳄": {"身长": 6.0, "体重": 550},
        "奥里诺科鳄": {"身长": 4.0, "体重": 225},
        "尼罗鳄": {"身长": 5.0, "体重": 300},
        "美洲鳄": {"身长": 5.5, "体重": 500},
        "盐水鳄": {"身长": 5.5, "体重": 400}
    },
    "蛇": {
        "眼镜王蛇": {"身长": 4.0, "体重": 8},
        "非洲岩蟒": {"身长": 7.0, "体重": 70},
        "网纹蟒": {"身长": 8.0, "体重": 60},
        "印度蟒": {"身长": 6.0, "体重": 50},
        "亚马逊绿水蟒": {"身长": 10.0, "体重": 270},
        "缅甸蟒": {"身长": 6.0, "体重": 90},
        "紫晶蟒": {"身长": 5.0, "体重": 100},
        "水蟒": {"身长": 8.0, "体重": 200}
    }
}

# 转换为DataFrame
records = []
for animal_type, animals in raw_data.items():
    for name, features in animals.items():
        records.append({
            "名称": name,
            "类别": animal_type,
            "身长": features["身长"],
            "体重": features["体重"]
        })

df = pd.DataFrame(records)
print("原始数据:")
print(df)

# ---------- 2. 数据预处理 ----------
# 标签分配：鳄鱼=1，蛇=-1
df['标签'] = df['类别'].map({'鳄鱼': 1, '蛇': -1})

# 特征提取与标准化（Z-Score归一化，解决尺度问题）
X = df[['身长', '体重']].values
y = df['标签'].values.reshape(-1, 1)

# 计算均值和标准差，标准化特征
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std  # 归一化后特征

print(f"\n数据形状: 原始X={X.shape}, 归一化X={X_normalized.shape}, y={y.shape}")
print(f"特征均值: {X_mean}, 特征标准差: {X_std}")

# ---------- 3. 初始化参数（优化初始化） ----------
np.random.seed(42)
w = np.random.randn(2, 1) * 0.1  # 稍大的初始化值（归一化后适配）
b = np.random.randn(1) * 0.1

print(f"\n初始参数:")
print(f"权重 w: {w.flatten()}")
print(f"偏置 b: {b[0]}")

# ---------- 4. 定义函数（优化损失和更新规则） ----------
def predict(X, w, b):
    """预测函数：线性得分 + 符号分类"""
    score = X @ w + b
    return np.sign(score), score

def compute_loss(X, y, w, b):
    """改进损失：合页损失（SVM Loss）+ L2正则化，更适合分类"""
    _, scores = predict(X, w, b)
    hinge_loss = np.mean(np.maximum(0, 1 - y * scores))  # 合页损失
    l2_reg = 0.01 * np.sum(w **2)  # L2正则化防止过拟合
    return hinge_loss + l2_reg

def update_parameters_gradient_descent(X, y, w, b, learning_rate=0.1):
    """梯度下降更新（带梯度计算，适配归一化数据）"""
    _, scores = predict(X, w, b)
    
    # 计算合页损失的梯度
    mask = (1 - y * scores) > 0  # 损失>0的样本
    dw = np.zeros_like(w)
    db = 0.0
    
    if np.any(mask):
        # 核心修复：将y[mask]重塑为列向量，保证形状匹配
        X_sub = X[mask.flatten()]
        y_sub = y[mask].reshape(-1, 1)  # 重塑为(m,1)，与X_sub(m,2)广播相乘
        # 合页损失梯度
        dw = -np.mean(X_sub * y_sub, axis=0).reshape(-1, 1)
        db = -np.mean(y_sub)
        # 加上L2正则化梯度
        dw += 2 * 0.01 * w
    
    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# ---------- 5. 训练模型 ----------
print("\n开始训练...")
learning_rate = 0.1
max_iterations = 2000
loss_history = []
accuracy_history = []

for iteration in range(max_iterations):
    # 计算损失
    current_loss = compute_loss(X_normalized, y, w, b)
    loss_history.append(current_loss)
    
    # 预测并计算准确率
    y_pred, _ = predict(X_normalized, w, b)
    accuracy = np.mean(y_pred == y)
    accuracy_history.append(accuracy)
    
    # 每200次迭代打印进度
    if iteration % 200 == 0:
        print(f"迭代 {iteration:4d} | 损失: {current_loss:.4f} | 准确率: {accuracy:.2%}")
    
    # 提前停止（准确率100%时）
    if accuracy == 1.0:
        print(f"提前停止于迭代 {iteration} (准确率100%)")
        break
    
    # 更新参数
    w, b = update_parameters_gradient_descent(X_normalized, y, w, b, learning_rate)

print("\n训练完成!")

# ---------- 6. 评估模型 ----------
# 原始特征的预测（反归一化）
def predict_original_scale(length, weight, w, b, X_mean, X_std):
    """对原始尺度特征预测"""
    x_normalized = (np.array([[length, weight]]) - X_mean) / X_std
    pred, _ = predict(x_normalized, w, b)
    return "鳄鱼" if pred == 1 else "蛇"

# 最终预测结果
y_pred, _ = predict(X_normalized, w, b)
df['预测'] = y_pred.flatten()
df['预测类别'] = df['预测'].map({1: '鳄鱼', -1: '蛇'})
df['是否正确'] = df['类别'] == df['预测类别']

print(f"\n最终结果:")
print(f"归一化权重 w: {w.flatten()}")
print(f"归一化偏置 b: {b[0]}")
print(f"最终准确率: {np.mean(df['是否正确']):.2%}")

print("\n分类详情:")
print(df[['名称', '类别', '身长', '体重', '预测类别', '是否正确']])

# 计算原始尺度的决策边界
# 归一化边界：w1*(L-L_mean)/L_std + w2*(W-W_mean)/W_std + b = 0
# 转换为原始尺度：W = (-w1*(L-L_mean)/L_std - b)*W_std/w2 + W_mean
def get_decision_boundary(L, w, b, X_mean, X_std):
    """计算原始尺度下，身长L对应的体重决策边界"""
    L_norm = (L - X_mean[0]) / X_std[0]
    W_norm = (-w[0]*L_norm - b) / w[1]
    W = W_norm * X_std[1] + X_mean[1]
    return W

print("\n决策边界（原始尺度）:")
# 决策边界方程推导
print(f"归一化空间: {w[0,0]:.4f}*(身长-{X_mean[0]:.2f})/{X_std[0]:.2f} + {w[1,0]:.4f}*(体重-{X_mean[1]:.2f})/{X_std[1]:.2f} + {b[0]:.4f} = 0")
print(f"原始空间: 体重 = {(-w[0,0]*X_std[1]/(w[1,0]*X_std[0])):.4f}*身长 + { (w[0,0]*X_mean[0]*X_std[1]/(w[1,0]*X_std[0]) - b[0]*X_std[1]/w[1,0] + X_mean[1]):.4f}")

# ---------- 7. 可视化 ----------
def plot_results(df, w, b, X_mean, X_std):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 子图1：数据+决策边界
    # 绘制数据点
    colors = {'鳄鱼': 'green', '蛇': 'red'}
    markers = {'鳄鱼': 'o', '蛇': 's'}
    for animal_type in ['鳄鱼', '蛇']:
        subset = df[df['类别'] == animal_type]
        ax1.scatter(subset['身长'], subset['体重'], 
                   c=colors[animal_type], marker=markers[animal_type], 
                   s=120, label=animal_type, edgecolors='k', alpha=0.8)
    
    # 绘制决策边界
    L_range = np.linspace(df['身长'].min()-0.5, df['身长'].max()+0.5, 100)
    W_boundary = get_decision_boundary(L_range, w, b, X_mean, X_std)
    ax1.plot(L_range, W_boundary, 'k--', linewidth=3, label='决策边界')
    
    # 标记错误点
    errors = df[~df['是否正确']]
    if not errors.empty:
        ax1.scatter(errors['身长'], errors['体重'], 
                   s=250, facecolors='none', edgecolors='blue', 
                   linewidths=2, label='错误分类')
    
    ax1.set_xlabel('身长 (米)', fontsize=12)
    ax1.set_ylabel('体重 (公斤)', fontsize=12)
    ax1.set_title('鳄鱼与蛇分类 - 决策边界', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：训练曲线
    ax2.plot(loss_history, 'b-', linewidth=2, label='损失值')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(accuracy_history, 'r-', linewidth=2, label='准确率')
    
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12, color='b')
    ax2_twin.set_ylabel('准确率', fontsize=12, color='r')
    ax2.set_title('训练过程 - 损失&准确率', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.tight_layout()
    plt.show()

plot_results(df, w, b, X_mean, X_std)

# ---------- 8. 新样本预测 ----------
print("\n新样本预测示例:")
test_samples = [
    (5.0, 250),  # 鳄鱼
    (7.0, 80),   # 蛇
    (4.5, 150),  # 边界样本
    (6.0, 300),  # 鳄鱼
    (9.0, 150),  # 蛇
]

for length, weight in test_samples:
    result = predict_original_scale(length, weight, w, b, X_mean, X_std)
    print(f"身长 {length}米, 体重 {weight}公斤 -> 预测为: {result}")