import tensorflow as tf

BATCH_SIZE = 3200

GPU_MEMORY_FRACTION = 0.8
# If you want to use 'dataset.shuffle' in 'get_image_dataset'
# tf.set_random_seed(123456)

def get_image_dataset_tfrecords(tfrecord_filename, batch_size=BATCH_SIZE, buffer_size=1000, resize=200):
    def _parse_function(example_proto):
        label_key_name = 'label'
        image_key_name = 'image_raw'
        features = {
            label_key_name: tf.FixedLenFeature((), tf.int64, default_value=0),
            image_key_name: tf.FixedLenFeature((), tf.string, default_value=''),
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image_decoded = tf.image.decode_jpeg(parsed_features[image_key_name], channels=3)
        if tf.shape(image_decoded) != (resize, resize, 3):
            image_decoded = tf.image.resize_images(image_decoded, (resize, resize))
        image_decoded.set_shape([resize, resize, 3])
        return image_decoded, parsed_features[label_key_name]

    # Creates a dataset that reads all of the examples from two files, and extracts
    # the image and label features.
    filenames = [tfrecord_filename]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    return dataset


def get_data(TFRECORD_FILENAME):

    dataset = get_image_dataset_tfrecords(TFRECORD_FILENAME, BATCH_SIZE)
    print(dataset)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION, allow_growth=True)
    with tf.device('/device:GPU:0'):
        # 1. Define your model, placeholders, loss, optimizer, and so on.

        # TODO

        with tf.Session(config=tf.ConfigProto(
            gpu_options = gpu_options,
            log_device_placement = False,
            allow_soft_placement = True
        )) as sess:
            elem = sess.run(next_element)
            print(elem)
            return elem

