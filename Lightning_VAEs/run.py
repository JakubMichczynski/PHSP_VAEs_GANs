from models.BetaVAE import Lit_BetaVAE
from models.MKMMD_VAE import Lit_MKMMD_VAE
from helpers.PhotonsDataModule import PhotonsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers



if __name__ == '__main__':
    # Hyperparameters
    RANDOM_SEED = 123
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 10000
    NUM_EPOCHS = 30
    LOGGING_INTERVAL = 300
    BETA_WEIGHT = 1
    SHUFFLE_TRAIN=True

    PLOT_FRACTION = 0.0125
    TEST_FRACTION = 0.0
    VALIDATION_FRACTION = 0.4
    NUM_WORKERS = 0
    DATA_PATH=None
    COLUMNS_KEYS=['E','X', 'Y', 'dX', 'dY', 'dZ']



    #Zmień nazwę scieżki
    MODEL_NAME='BetaVAE'
    SAVE_MODEL_DIR_PATH = f'/checkpoints/{MODEL_NAME}/'

    LOAD_CHECKPOINT_PATH = f'/checkpoints/MKMMD_VAE/lasy.ckpt'


    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="/home/jakmic/Projekty/dose3d-phsp/AE_VAE/Lighting_Autoencoders/results/", name="BetaVAE")
    csv_logger = pl_loggers.CSVLogger(save_dir="/home/jakmic/Projekty/dose3d-phsp/AE_VAE/Lighting_Autoencoders/results/", name=MODEL_NAME)

    dm=PhotonsDataModule(data_path=DATA_PATH,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,test_fraction=TEST_FRACTION,validation_fraction=VALIDATION_FRACTION,shuffle_train=SHUFFLE_TRAIN,random_seed=RANDOM_SEED, columns_keys=COLUMNS_KEYS)
    
    model = Lit_BetaVAE(beta_weight=BETA_WEIGHT, learning_rate=LEARNING_RATE)
    # model =Lit_MKMMD_VAE(learning_rate=LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(dirpath=SAVE_MODEL_DIR_PATH, filename='checkpoint'+f'_{MODEL_NAME}_'+'{epoch}epoch', auto_insert_metric_name=False, save_last=True)

    trainer = Trainer(gpus=-1,logger=csv_logger, callbacks=[checkpoint_callback],max_epochs=NUM_EPOCHS, fast_dev_run=False, log_every_n_steps=1)

    trainer.fit(model, dm)
    # trainer.fit(model, dm, ckpt_path=LOAD_CHECKPOINT_PATH)