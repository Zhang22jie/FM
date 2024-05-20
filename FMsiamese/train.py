import datetime
import os
import random
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam

from nets.siamese import siamese
from utils.callbacks import LossHistory, ModelCheckpoint
from utils.dataloader import Datasets
from utils.utils import get_lr_scheduler, show_config

def train_siamese(train_lines, train_labels, val_lines, val_labels,b ):   # 训练函数
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    # train_own_data = True
    train_gpu = [0, ]
    input_shape = [105, 105]   # 输入为105*105
    Init_Epoch = 0    # 初始化
    Epoch = 200      # 训练轮数
    batch_size = 32  # 批大小
    Init_lr = 1e-2  # 初始化的学习率
    Min_lr = Init_lr * 0.01  # 最小学习率
    optimizer_type = "sgd"  # 优化器类型
    momentum = 0.9  # 动量参数，加速优化过程
    save_period = 10  # 模型保存周期
    lr_decay_type = 'cos'  # 学习率衰减类型
    save_dir = 'logs/diflogs/' + str(b) + '/'  # 保存模型权重的路径
    num_workers = 1  # 数据加载的子进程数量
    model_path = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"  # 模型权重
    # 设置要用到的显卡
    # ------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # ------------------------------------------------------#
    #   判断当前使用的GPU数量与机器上实际的GPU数量
    # ------------------------------------------------------#
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    if ngpus_per_node > 1:
        with strategy.scope():
            model = siamese(input_shape=[input_shape[0], input_shape[1], 3])
            if model_path != '':
                # ------------------------------------------------------#
                #   载入预训练权重
                # ------------------------------------------------------#
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        model = siamese(input_shape=[input_shape[0], input_shape[1], 3])
        if model_path != '':
            # ------------------------------------------------------#
            #   载入预训练权重
            # ------------------------------------------------------#
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # 计算训练集和验证集的样本数量
    num_train = len(train_lines)
    num_val = len(val_lines)
    # 模型配置信息
    show_config(
        model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )
    # ---------------------------------------------------------#
    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4  # 根据优化器类型设定的一个推荐步数
    total_step = num_train // batch_size * Epoch  # 总训练步数
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // batch_size) + 1  # 如果训练步数太少，下面是警告
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, batch_size, Epoch, total_step))
        print(
            "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))
    # -------------------------------------------------------------#
    #   训练分为两个阶段，两阶段初始的学习率不同，手动调节了学习率
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    # -------------------------------------------------------------#
    if True:
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率，其中损失函数使用二元交叉熵
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max) # 初始学习率
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2) # 最小学习率

        optimizer = {
            'adam': Adam(lr=Init_lr_fit, beta_1=momentum),
            'sgd': SGD(lr=Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]   # 优化器设置
        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        train_dataset = Datasets(input_shape, train_lines, train_labels, batch_size, True)
        val_dataset = Datasets(input_shape, val_lines, val_labels, batch_size, False)
        if ngpus_per_node > 1:
            with strategy.scope():
                model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
        else:
            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
        # -------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler       用于设置学习率下降的方式
        #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
        # -------------------------------------------------------------------------------#
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        logging = TensorBoard(log_dir)
        loss_history = LossHistory(log_dir)
        checkpoint = ModelCheckpoint(
            os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
            monitor='val_loss', save_weights_only=True, save_best_only=False, period=save_period)
        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=False,
                                          period=1)
        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
        callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        History = model.fit(
            x=train_dataset,
            steps_per_epoch=epoch_step,
            validation_data=val_dataset,
            validation_steps=epoch_step_val,
            epochs=Epoch,
            initial_epoch=Init_Epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=callbacks
        )
        # 保存数据结果
        train_loss = History.history['loss']
        binary_accuracy = History.history['binary_accuracy']
        val_loss = History.history['val_loss']
        val_binary_accuracy = History.history['val_binary_accuracy']
        np.save('./outputs/' + str(b) + '/train_loss.npy', train_loss)
        np.save('./outputs/' + str(b) + '/binary_accuracy.npy', binary_accuracy)
        np.save('./outputs/' + str(b) + '/val_loss.npy', val_loss)
        np.save('./outputs/' + str(b) + '/val_binary_accuracy.npy', val_binary_accuracy)
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.plot(range(1, Epoch + 1), train_loss, label='train_loss')
        plt.plot(range(1, Epoch + 1), val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(122)
        plt.plot(range(1, Epoch + 1), binary_accuracy, label='binary_accuracy')
        plt.plot(range(1, Epoch + 1), val_binary_accuracy, label='val_binary_accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.savefig('./outputs/' + str(b) + '/loss_acc.png')
        # plt.show()
        print('结束第一轮训练')
    # return model, History

# Example usage:
# 训练不同台站数据的权重模型
if __name__ == "__main__":
    dataset_path = "datasetss" # 台站数据文件存放，datasetss存在文件夹image
    b = 8   # 相对应的训练的台站
    characters_to_use = ['6', '8', '10', '12', '14', '16', '18' '20']  # 初始化字符列表 image下的不同信号文件
    for i in range(0, len(characters_to_use), 1):  # 每次训练两个字符
        train_characters = characters_to_use[i:i + 1]
        new_character = '00'  # 示例字符串  这个是image文件夹下存放的噪声数据
        train_characters.append(new_character)  # 增加噪声数据
        types = 0
        train_path = os.path.join(dataset_path, 'images')
        train_ratio = 0.9  # 划分训练集和验证集的比例
        lines = []
        labels = []
        b = b + 1
        for character in train_characters:  # 对传入的数据进行训练
            # -------------------------------------------------------------#
            #   对每张图片进行遍历
            # -------------------------------------------------------------#
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                print(os.path.join(character_path, image))
                lines.append(os.path.join(character_path, image))
                labels.append(types)
            types += 1
        # -------------------------------------------------------------#
        #   将获得的所有图像进行打乱获得数据标签。
        # -------------------------------------------------------------#
        random.seed(1)
        shuffle_index = np.arange(len(lines), dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        lines = np.array(lines, dtype=np.object)
        labels = np.array(labels)
        lines = lines[shuffle_index]
        labels = labels[shuffle_index]
        # -------------------------------------------------------------#
        #   将训练集和验证集进行划分
        # -------------------------------------------------------------#
        num_train = int(len(lines) * train_ratio)
        val_lines = lines[num_train:]
        val_labels = labels[num_train:]
        train_lines = lines[:num_train]
        train_labels = labels[:num_train]
        train_siamese(train_lines, train_labels, val_lines, val_labels, b)  # 调用训练函数进行模型训练


