
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"

NUM_CLASSES = len(CHAR_VECTOR) + 1

#icdar
FLAGS = {'training_data_path': 'C:/Users/Raidas/Desktop/fots/ICDAR15/ch4_training_images',
         'training_annotation_path': 'C:/Users/Raidas/Desktop/fots/ICDAR15/ch4_training_localization_transcription_gt',
         'min_text_size': 0}

#synthtext
FLAGS = {'training_data_path': 'D:/data/SynthText/SynthText/',
         'training_annotation_path': '',
         'min_text_size': 0}
