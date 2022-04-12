import os
from absl import app
from absl import flags
import imageio
import numpy as np
import tensorflow as tf
from NccFlow.data_conversion_scripts import conversion_utils

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Dataset folder.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_integer('shard', 0, 'Which shard this is.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')

def convert_dataset():
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.mkdir(FLAGS.output_dir)

    data_split = 'training'
    if data_split == 'training':
        split_folder = os.path.join(FLAGS.output_dir, data_split)
        if not tf.io.gfile.exists(split_folder):
            tf.io.gfile.mkdir(split_folder)
        for data_type in ['clean', 'final']:
            output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
            if not tf.io.gfile.exists(output_folder):
                tf.io.gfile.mkdir(output_folder)
                output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
                input_folder = os.path.join(FLAGS.data_dir, data_split, data_type)
                flow_folder = os.path.join(FLAGS.data_dir, data_split, 'flow')
                occlusion_folder = os.path.join(FLAGS.data_dir, data_split, 'occlusions')

                image_folders = sorted(tf.io.gfile.glob(input_folder + '/*'))

                occlusion_folders = sorted(tf.io.gfile.glob(occlusion_folder + '/*'))
                flow_folders = sorted(tf.io.gfile.glob(flow_folder + '/*'))
                assert len(image_folders) == len(flow_folders)
                assert len(flow_folders) == len(occlusion_folders)

                data_list = []

                for image_folder, flow_folder, occlusion_folder in zip(
                        image_folders, flow_folders, occlusion_folders):
                    images = tf.io.gfile.glob(image_folder + '/*png')

                    sort_by_frame_index = lambda x: int(
                        os.path.basename(x).split('_')[1].split('.')[0])
                    images = sorted(images, key=sort_by_frame_index)

                    flows = tf.io.gfile.glob(flow_folder + '/*flo')
                    flows = sorted(flows, key=sort_by_frame_index)
                    occlusions = tf.io.gfile.glob(occlusion_folder + '/*png')
                    occlusions = sorted(occlusions, key=sort_by_frame_index)

                    image_pairs = zip(images[:-1], images[1:])

                    assert len(flows) == len(images) - 1 == len(occlusions)
                    data_list.extend(zip(image_pairs, flows, occlusions))

                write_records(data_list, output_folder)

def write_records(data_list, output_folder):
    tmpdir = '/tmp/flying_chairs'
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    filenames = conversion_utils.generate_sharded_filenames(
        os.path.join(output_folder, 'sintel@{}'.format(FLAGS.num_shards)))
    with tf.io.TFRecordWriter(filenames[FLAGS.shard]) as record_writer:
        total = len(data_list)
        images_per_shard = total // FLAGS.num_shards
        start = images_per_shard * FLAGS.shard
        end = start + images_per_shard
        if FLAGS.shard == FLAGS.num_shards - 1:
            data_list = data_list[start:]
        else:
            data_list = data_list[start:end]
        tf.compat.v1.logging.info('Writing %d images per shard', images_per_shard)
        tf.compat.v1.logging.info('Writing range %d to %d of %d total.', start, end,
                                  total)
        img1_path = os.path.join(tmpdir, 'img1.png')
        img2_path = os.path.join(tmpdir, 'img2.png')
        flow_path = os.path.join(tmpdir, 'flow.flo')
        occlusion_path = os.path.join(tmpdir, 'occlusion.png')

        for i, (images, flow, occlusion, invalids) in enumerate(data_list):
            # 逐步删除数据
            if os.path.exists(img1_path):
                os.remove(img1_path)
            if os.path.exists(img2_path):
                os.remove(img2_path)
            if os.path.exists(flow_path):
                os.remove(flow_path)
            if os.path.exists(occlusion_path):
                os.remove(occlusion_path)

            # 进行一个文件读取的工作
            tf.io.gfile.copy(images[0], img1_path)
            tf.io.gfile.copy(images[1], img2_path)
            # 读取图像数据
            image1_data = imageio.imread(img1_path)
            image2_data = imageio.imread(img2_path)

            if flow is not None:
                assert occlusion is not None
                tf.io.gfile.copy(flow, flow_path)
                tf.io.gfile.copy(occlusion, occlusion_path)
                flow_data = conversion_utils.read_flow(flow_path)
                occlusion_data = np.expand_dims(
                    imageio.imread(occlusion_path) // 255, axis=-1)

                height = image1_data.shape[0]
                width = image1_data.shape[1]
                assert height == image2_data.shape[0] == flow_data.shape[0]
                assert width == image2_data.shape[1] == flow_data.shape[1]
                assert height == occlusion_data.shape[0]
                assert width == occlusion_data.shape[1]

                feature = {
                    'height': conversion_utils.int64_feature(height),
                    'width': conversion_utils.int64_feature(width),
                    'image1_path': conversion_utils.bytes_feature(str.encode(images[0])),
                    'image2_path': conversion_utils.bytes_feature(str.encode(images[1])),
                }
                if flow is not None:
                    feature.update({
                        'flow_uv':
                            conversion_utils.bytes_feature(flow_data.tobytes()),
                        'occlusion_mask':
                            conversion_utils.bytes_feature(occlusion_data.tobytes()),
                        'flow_path':
                            conversion_utils.bytes_feature(str.encode(flow)),
                        'occlusion_path':
                            conversion_utils.bytes_feature(str.encode(occlusion)),
                    })
                example = tf.train.SequenceExample(
                    context=tf.train.Features(feature=feature),
                    feature_lists=tf.train.FeatureLists(
                        feature_list={
                            'images':
                                tf.train.FeatureList(feature=[
                                    conversion_utils.bytes_feature(image1_data.tobytes()),
                                    conversion_utils.bytes_feature(image2_data.tobytes())
                                ])
                        }))
                if i % 10 == 0:
                    tf.compat.v1.logging.info('Writing %d out of %d total.', i,
                                              len(data_list))
                record_writer.write(example.SerializeToString())

    tf.compat.v1.logging.info('Saved results to %s', FLAGS.output_dir)

def main(_):
  convert_dataset()

if __name__ == '__main__':
  app.run(main)