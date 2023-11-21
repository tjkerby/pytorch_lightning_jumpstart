conf = dict(
    lr = 5e-4,
    bs = 32,
    num_workers = 62,
    hidden_size = 100,
    drop_out_p = .2,
    save_path = './checkpoints/save_path_name/',
    patience = 20,
    logger = dict(
        version = 'version_name'
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