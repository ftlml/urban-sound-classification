import os

CLASSES = [
    'air conditioner',
    'car horn',
    'children playing',
    'dog bark',
    'drilling',
    'engine idling',
    'gun shot',
    'jackhammer',
    'siren',
    'street music'
]

CLASSES_DICTIONARY = {
    CLASSES[idx]: idx for idx in range(len(CLASSES))
}

CLASSES_REVERSE_DICTIONARY = dict(
    zip(CLASSES_DICTIONARY.values(), CLASSES_DICTIONARY.keys())
)

SAMPLE_RATE = 22050

MAX_SECS = 4

MAX_SAMPLES = SAMPLE_RATE * MAX_SECS

SOUND_FILE_DIRS = [
  'Data/UrbanSound8K/audio/fold1',
  'Data/UrbanSound8K/audio/fold2',
  'Data/UrbanSound8K/audio/fold3',
  'Data/UrbanSound8K/audio/fold4',
  'Data/UrbanSound8K/audio/fold5',
  'Data/UrbanSound8K/audio/fold6',
  'Data/UrbanSound8K/audio/fold7',
  'Data/UrbanSound8K/audio/fold8',
  'Data/UrbanSound8K/audio/fold9',
  'Data/UrbanSound8K/audio/fold10'
]

SOUND_FILE_PATHS = filter(
  lambda FP: FP if FP.endswith('.wav') else None, [
  os.path.join(DIR, FP) \
  for DIR in SOUND_FILE_DIRS \
  for FP in os.listdir(DIR)
])

SPECGRAM_SHAPE = (129, 689)

TF_RECORDS_DESTINATION = 'Data'
TF_RECORDS_META = os.path.join(
  TF_RECORDS_DESTINATION, 'urban_sound.meta'
)
TF_RECORDS_TRAIN = os.path.join(
  TF_RECORDS_DESTINATION, 'urban_sound_train.tfrecords'
)
TF_RECORDS_TEST = os.path.join(
  TF_RECORDS_DESTINATION, 'urban_sound_test.tfrecords'
)
