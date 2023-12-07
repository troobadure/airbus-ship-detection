BASE_DIR = '' # 'airbus-ship-detection/'
TRAIN_DIR = BASE_DIR + 'input/train_v2'
TEST_DIR = BASE_DIR + 'input/test_v2'
MASKS_PATH = BASE_DIR + 'input/train_ship_segmentations_v2.csv'

BATCH_SIZE = 8
EDGE_CROP = 16

# downsampling in preprocessing
IMG_SCALING = (3, 3)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# number of validation images to use
VALID_IMG_COUNT = 100

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 16
MAX_TRAIN_EPOCHS = 10
AUGMENT_BRIGHTNESS = False

BALANCE_SHIP_COUNT = False
EMPTY_DROP_COUNT = 70000

#TODO: sort out the constants