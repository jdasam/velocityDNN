from __future__ import division

class Config:
    def __init__(self, name='results'):

        # data
        self.data_dir = 'input/16000_2048'
        self.del_duplicate = True
        self.seg_length = 100
        self.add_diff = False
        self.feature_names = ['spec']
        self.feature_dim = 39

        # Network
        self.type = 'cnn_type_1'
        self.chroma = False
        self.onset_weight = True
        self.onset = False
        self.output_dim = 43

        # Training / Test Parameter
        self.batch_size = 128
        self.learning_rate = 0.1
        self.validation_per_epoch = 1
        self.optimizer = 'adam'  # 'adam'
        self.lr_drop_multiplier = 1/5
        self.n_lr_drop = 1
        self.patients = 5
        # self.patients_threshold = 0
        self.monitor = 'acc'  # 'loss'
        self.dropout = [0.5, 0.5]
        self.buffer_size = 20
        self.regularize = 1e-7

        # misc
        self.name = name
        self.fold = 1
        self.debug = False
        self.max_save = 1
        self.visible_gpu = 0
        self.save_dir = None
        self.verbose = 1

    def save_dir_generate(self):
        levels = []
        level_1 = self.name
        levels.append(level_1)

        input_type = self.data_dir.split('/')[-1]
        if self.add_diff:
            input_type + '_diff'
        level_2 = input_type
        levels.append(level_2)

        model_type = self.type
        level_3 = model_type
        levels.append(level_3)

        optimize_type = '{}_{:.3e}'.format(self.optimizer, self.learning_rate)
        level_4 = optimize_type
        levels.append(level_4)

        level_5 = 'fold_{:d}'.format(self.fold)
        levels.append(level_5)

        dir_name = '/'.join(levels)
        self.save_dir = dir_name
        return dir_name
