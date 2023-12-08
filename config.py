BASE_DIR = '' # 'airbus-ship-detection/'
TRAIN_DIR = BASE_DIR + 'input/train_v2'
TEST_DIR = BASE_DIR + 'input/test_v2'
MASKS_PATH = BASE_DIR + 'input/train_ship_segmentations_v2.csv'

BEST_MODEL_PATH="model_checkpoint.best.hdf5"
FULLRES_MODEL_PATH="fullres_model.h5"
LOAD_MODEL_PATH = 'fullres_model.h5'

IMG_SCALING = (3, 3)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

BATCH_SIZE = 32
MAX_TRAIN_EPOCHS = 100
MAX_TRAIN_STEPS = 100
MAX_VAL_STEPS = 5

REMOVE_CORRUPT = False
BALANCE_SHIP_COUNT = True
SAMPLES_PER_GROUP = 2000
EMPTY_DROP_COUNT = 150000
AUGMENT_BRIGHTNESS = False