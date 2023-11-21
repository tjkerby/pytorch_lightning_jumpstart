conf = dict(
    lr = 5e-4,
    bs = 32,
    num_workers = 62,
    hidden_size = 100,
    drop_out_p = .0,
    save_path = './checkpoints/local_corex_mlp_mnist/no_dropout/',
    patience = 20,
    logger = dict(
        # name = 'tb_logs',
        version = 'bs_128_CELoss_lr_5em4_do_0_hs_100'
    ),
    trainer = dict(
        devices = 4,
        max_epochs = 200,
        accelerator = 'gpu',
        strategy = 'ddp',
        default_root_dir = '/usr/src/app',#'./', #
        refresh_rate = 50
    )
)