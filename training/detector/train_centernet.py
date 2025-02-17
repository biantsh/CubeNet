import argparse
import glob
import json
import os
import shutil

import tensorflow as tf
from google.protobuf import text_format
from object_detection import model_lib_v2, export_tflite_graph_lib_tf2
from object_detection.protos import pipeline_pb2
from tensorflow.lite.python import lite


def main(dataset_dir: str, output_dir: str, num_steps: int) -> None:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(current_dir, 'center_net')

    checkpoint_path = os.path.join(model_dir, 'checkpoint', 'ckpt-0')
    pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
    label_map_path = os.path.join(model_dir, 'label_map.pbtxt')

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    config_path_output = os.path.join(output_dir, 'pipeline.config')

    saved_model_dir = os.path.join(output_dir, 'saved_model')
    tflite_path = os.path.join(output_dir, 'model.tflite')

    train_path = os.path.join(dataset_dir, 'train.record-?????-of-00100')
    test_path = os.path.join(dataset_dir, 'test.record-?????-of-00050')

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as config_file:
        text_format.Merge(config_file.read(), pipeline_config)

    train_reader = pipeline_config.train_input_reader
    train_reader.tf_record_input_reader.input_path[:] = [train_path]
    train_reader.label_map_path = label_map_path

    test_reader = pipeline_config.eval_input_reader[0]
    test_reader.tf_record_input_reader.input_path[:] = [test_path]
    test_reader.label_map_path = label_map_path

    pipeline_config.train_config.fine_tune_checkpoint = checkpoint_path
    with tf.io.gfile.GFile(config_path_output, 'wb') as config_file:
        config_file.write(text_format.MessageToString(pipeline_config))

    with tf.compat.v2.distribute.MirroredStrategy().scope():
        model_lib_v2.train_loop(
            pipeline_config_path=config_path_output,
            train_steps=num_steps,
            model_dir=output_dir
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    for file_path in glob.glob(f'{output_dir}/*'):
        file_name = os.path.basename(file_path)

        if file_name == 'checkpoint' or file_name.startswith('ckpt-'):
            shutil.move(file_path, checkpoint_dir)

    model_lib_v2.eval_continuously(
        pipeline_config_path=config_path_output,
        checkpoint_dir=checkpoint_dir,
        model_dir=output_dir,
        timeout=0
    )

    export_tflite_graph_lib_tf2.export_tflite_model(
        trained_checkpoint_dir=checkpoint_dir,
        pipeline_config=pipeline_config,
        output_directory=output_dir,
        include_keypoints=False,
        use_regular_nms=False,
        max_detections=1
    )

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as tflite_file:
        tflite_file.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--num_steps', type=int, default=5000)

    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir,  args.num_steps)
