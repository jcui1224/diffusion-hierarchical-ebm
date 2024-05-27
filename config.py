a100_celeba256 = {
    'dataset': 'celeba_256',
    'batch_size': 2,
    'e_l_steps': 30,
    'e_l_step_size': 2e-5,
    'ch': 64,
    'num_res_blocks': 4,
    'e_lr': 4e-5,
    'wd': 3e-5,
    'e_gamma': 0.98,
    'bn': False,
    'data_base_dir':  '/Tian-ds/jcui7/HugeData/',
    'ckpt_base_dir': '/Tian-ds/jcui7/NVAE/ckpt/eval-.',
    'fid_stat_base_dir': '/Tian-ds/jcui7/HugeData/fid_stats/celeba256/fid_stats_celeba256_train.npz'
}

a100_cifar10 = {
    'dataset': 'cifar10',
    'batch_size': 25,
    'e_l_steps': 50,
    'e_l_step_size': 3e-5,
    'ch': 128,
    'num_res_blocks': 4,
    'e_lr': 4e-5,
    'wd': 3e-5,
    'e_gamma': 0.9,
    'bn': True,
    'data_base_dir': '/Tian-ds/jcui7/HugeData/',
    'ckpt_base_dir': '/Tian-ds/jcui7/NVAE/ckpt/eval-.',
    'fid_stat_base_dir': '/Tian-ds/jcui7/HugeData/fid_stats/cifar10/fid_stats_cifar10_train.npz'
}

