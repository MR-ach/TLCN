import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import tensorflow as tf
from keras import layers, Model
import pickle
import time

# 设置随机种子
seed = 600
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 数据加载函数
def load_labeled_data(names):
    datasets = []
    labels = []
    for label_idx, name in enumerate(names):
        with open(f'../data/test/{name}.pkl', 'rb') as f:
            data = pickle.load(f)
            data = data.reshape(-1, 3072, 1)
            num_samples = data.shape[0]
            file_labels = np.full(num_samples, label_idx)
            datasets.append(data)
            labels.append(file_labels)
    X = np.concatenate(datasets, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


# 加载数据
names = ['Z0', 'Z1', 'Z2', 'Z3', 'Z4']
X, y = load_labeled_data(names)
print("数据形状:", X.shape)
print("标签分布:", np.unique(y, return_counts=True))

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.666, stratify=y_temp)

# 转换为分类格式
num_classes = len(names)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# ========== 新增LoRA卷积层 ==========
class LoRA_Conv1D(layers.Layer):
    """低秩自适应卷积层"""

    def __init__(self, filters, kernel_size, lora_rank=8, strides=1, padding='valid', dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.lora_rank = lora_rank
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]

        # 原始卷积核参数（冻结）
        self.kernel = self.add_weight(
            name='kernel',
            shape=(kernel_size, input_dim, self.filters),
            initializer='glorot_uniform',
            trainable=False  # 冻结原始参数
        )

        # 低秩适配参数
        self.lora_A = self.add_weight(
            name='lora_A',
            shape=(kernel_size * input_dim, self.lora_rank),
            initializer='glorot_uniform',
            trainable=True
        )
        self.lora_B = self.add_weight(
            name='lora_B',
            shape=(self.lora_rank, self.filters),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # 计算低秩更新
        kernel_size = self.kernel.shape[0]
        input_dim = self.kernel.shape[1]

        # 将原始核和低秩更新转换为矩阵形式
        original_kernel = tf.reshape(self.kernel, [-1, self.filters])
        delta_kernel = tf.matmul(self.lora_A, self.lora_B)

        # 组合更新后的核
        updated_kernel = tf.reshape(original_kernel + delta_kernel,
                                    [kernel_size, input_dim, self.filters])

        # 执行卷积操作
        return tf.nn.conv1d(
            inputs,
            updated_kernel,
            stride=self.strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
            data_format='NWC'
        )


# ========== 修改后的模型组件 ==========
class WaveConvNeXtBlock(layers.Layer):
    def __init__(self, dim, expansion_rate=4, dilation_rate=1, drop_path_rate=0.0):
        super().__init__()
        # 深度卷积部分（分解为depthwise + pointwise）
        self.depthwise = layers.DepthwiseConv1D(
            kernel_size=5,
            padding='same',
            dilation_rate=dilation_rate,
            depth_multiplier=1
        )
        self.pointwise = LoRA_Conv1D(  # 对pointwise卷积应用LoRA
            filters=dim,
            kernel_size=1,
            padding='same',
            lora_rank=8
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

        # 恢复为标准全连接层
        self.pwconv1 = layers.Dense(dim * expansion_rate)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)

        self.drop_path = layers.Dropout(
            drop_path_rate, noise_shape=(None, 1, 1)) if drop_path_rate > 0.0 else lambda x: x
        self.dim = dim

    def build(self, input_shape):
        # 维度投影恢复为标准全连接
        if input_shape[-1] != self.dim:
            self.proj = layers.Dense(self.dim)
        else:
            self.proj = lambda x: x
        super().build(input_shape)

    def call(self, inputs):
        shortcut = self.proj(inputs)
        x = self.depthwise(inputs)
        x = self.pointwise(x)  # LoRA卷积
        x = self.norm(x)
        x = self.pwconv1(x)  # 标准全连接
        x = self.act(x)
        x = self.pwconv2(x)  # 标准全连接
        return shortcut + self.drop_path(x)


class WaveConvNeXt(Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # self.input_layer = layers.Input(shape=input_shape)
        # 输入预处理（使用LoRA卷积）
        self.stem = tf.keras.Sequential([
            LoRA_Conv1D(32, 5, strides=2, padding='same', lora_rank=8, input_shape=input_shape),
            layers.BatchNormalization()
        ])

        # 分阶段结构（采样使用LoRA卷积）
        self.stages = tf.keras.Sequential([
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(32, dilation_rate=2 ** (i % 4)) for i in range(2)],
                LoRA_Conv1D(16, 2, strides=2, padding='valid', lora_rank=8),
                layers.BatchNormalization()
            ]),
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(16, dilation_rate=2 ** (i % 4)) for i in range(2)],
                LoRA_Conv1D(8, 2, strides=2, padding='valid', lora_rank=8),
                layers.BatchNormalization()
            ]),
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(8, dilation_rate=2 ** (i % 4)) for i in range(4)],
                LoRA_Conv1D(4, 2, strides=2, padding='valid', lora_rank=8),
                layers.BatchNormalization()
            ]),
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(16, dilation_rate=2 ** (i % 4)) for i in range(4)],
                LoRA_Conv1D(4, 2, strides=2, padding='valid'),
                layers.BatchNormalization()
            ]),
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(32, dilation_rate=2 ** (i % 4)) for i in range(4)],
                LoRA_Conv1D(4, 2, strides=2, padding='valid'),
                layers.BatchNormalization()
            ]),
            tf.keras.Sequential([
                *[WaveConvNeXtBlock(4, dilation_rate=2 ** (i % 4)) for i in range(2)]
            ])
        ])

        # 分类头恢复为标准全连接
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes)
        self.softmax = layers.Activation('softmax')

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.stages(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return self.softmax(x)


# ========== 模型初始化与训练 ==========
model = WaveConvNeXt(X_train.shape[1:], num_classes)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.build(X_train.shape)
model.summary()

# 训练模型
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32
)


# ========== 评估与可视化 ==========
def comprehensive_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class, digits=4))

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred_class),
                annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix")
    plt.savefig('../result/Private/WaveConvNeXt/lora_confusion_matrix.png')
    plt.show()

    # 推理速度测试
    start_time = time.time()
    model.predict(X_test[:100])
    end_time = time.time()
    print(f"Inference time per sample: {(end_time - start_time) / 100:.6f} sec")

    # 噪声鲁棒性测试
    noise_levels = [0.1, 0.2, 0.3]
    for noise in noise_levels:
        noisy_X = X_test + np.random.normal(0, noise, X_test.shape)
        _, acc = model.evaluate(noisy_X, y_test, verbose=0)
        print(f"Noise level {noise}: Test accuracy = {acc:.4f}")

print("\n=== 综合评估结果 ===")
comprehensive_evaluation(model, X_test, y_test)

# 训练过程可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('../result/Private/WaveConvNeXt/lora_training_history_plot.png')
plt.show()

# 保存训练历史
history_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Train Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy'],
    'Train Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
history_df.to_excel('../result/Private/WaveConvNeXt/lora_training_history.xlsx', index=False)
print("Training history has been saved.")