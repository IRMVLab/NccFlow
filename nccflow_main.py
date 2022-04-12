# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to train and evaluate NccFlow."""

# pylint:disable=g-importing-member
from functools import partial
from absl import app
from absl import flags

import gin
import numpy as np
import tensorflow as tf

from NccFlow import augmentation
from NccFlow import data
# pylint:disable=unused-import
from NccFlow import flags
from NccFlow import plotting
from NccFlow.net import NccFlow

FLAGS = flags.FLAGS

import os
import time


def create_nccflow():
  """Build the nccflow model."""

  build_selfsup_transformations = partial(
      augmentation.build_selfsup_transformations,
      crop_height=FLAGS.selfsup_crop_height,
      crop_width=FLAGS.selfsup_crop_width,
      max_shift_height=FLAGS.selfsup_max_shift,
      max_shift_width=FLAGS.selfsup_max_shift,
      resize=FLAGS.resize_selfsup)

  # Define learning rate schedules [none, cosine, linear, expoential].
  def learning_rate_fn():
    step = tf.compat.v1.train.get_or_create_global_step()
    effective_step = tf.maximum(step - FLAGS.lr_decay_after_num_steps + 1, 0)
    lr_step_ratio = tf.cast(effective_step, 'float32') / float(
        FLAGS.lr_decay_steps)
    if FLAGS.lr_decay_type == 'none' or FLAGS.lr_decay_steps <= 0:
      return FLAGS.gpu_learning_rate
    elif FLAGS.lr_decay_type == 'cosine':
      x = np.pi * tf.minimum(lr_step_ratio, 1.0)
      return FLAGS.gpu_learning_rate * (tf.cos(x) + 1.0) / 2.0
    elif FLAGS.lr_decay_type == 'linear':
      return FLAGS.gpu_learning_rate * tf.maximum(1.0 - lr_step_ratio, 0.0)
    elif FLAGS.lr_decay_type == 'exponential':
      return FLAGS.gpu_learning_rate * 0.5**lr_step_ratio
    else:
      raise ValueError('Unknown lr_decay_type', FLAGS.lr_decay_type)

  occ_weights = {
      'fb_abs': FLAGS.occ_weights_fb_abs,
      'forward_collision': FLAGS.occ_weights_forward_collision,
      'backward_zero': FLAGS.occ_weights_backward_zero,
  }
  # Switch off loss-terms that have weights < 1e-2.
  occ_weights = {k: v for (k, v) in occ_weights.items() if v > 1e-2}

  occ_thresholds = {
      'fb_abs': FLAGS.occ_thresholds_fb_abs,
      'forward_collision': FLAGS.occ_thresholds_forward_collision,
      'backward_zero': FLAGS.occ_thresholds_backward_zero,
  }
  occ_clip_max = {
      'fb_abs': FLAGS.occ_clip_max_fb_abs,
      'forward_collision': FLAGS.occ_clip_max_forward_collision,
  }

  nccflow = NccFlow(
      checkpoint_dir=FLAGS.checkpoint_dir,
      checkpoint_dir1=FLAGS.checkpoint_dir1,
      checkpoint_dir2=FLAGS.checkpoint_dir2,
      optimizer=FLAGS.optimizer,
      learning_rate=learning_rate_fn,
      only_forward=FLAGS.only_forward,
      level1_num_layers=FLAGS.level1_num_layers,
      level1_num_filters=FLAGS.level1_num_filters,
      level1_num_1x1=FLAGS.level1_num_1x1,
      dropout_rate=FLAGS.dropout_rate,
      build_selfsup_transformations=build_selfsup_transformations,
      fb_sigma_teacher=FLAGS.fb_sigma_teacher,
      fb_sigma_student=FLAGS.fb_sigma_student,
      train_with_supervision=FLAGS.use_supervision,
      train_with_gt_occlusions=FLAGS.use_gt_occlusions,
      smoothness_edge_weighting=FLAGS.smoothness_edge_weighting,
      teacher_image_version=FLAGS.teacher_image_version,
      stop_gradient_mask=FLAGS.stop_gradient_mask,
      selfsup_mask=FLAGS.selfsup_mask,
      normalize_before_cost_volume=FLAGS.normalize_before_cost_volume,
      original_layer_sizes=FLAGS.original_layer_sizes,
      shared_flow_decoder=FLAGS.shared_flow_decoder,
      channel_multiplier=FLAGS.channel_multiplier,
      num_levels=FLAGS.num_levels,
      use_cost_volume=FLAGS.use_cost_volume,
      use_feature_warp=FLAGS.use_feature_warp,
      accumulate_flow=FLAGS.accumulate_flow,
      occlusion_estimation=FLAGS.occlusion_estimation,
      occ_weights=occ_weights,
      occ_thresholds=occ_thresholds,
      occ_clip_max=occ_clip_max,
      smoothness_at_level=FLAGS.smoothness_at_level,
  )
  return nccflow


def check_model_frozen(feature_model, flow_model, prev_flow_output=None):
  """Check that a frozen model isn't somehow changing over time."""
  state = np.random.RandomState(40)
  input1 = state.randn(FLAGS.batch_size, FLAGS.height, FLAGS.width,
                       3).astype(np.float32)
  input2 = state.randn(FLAGS.batch_size, FLAGS.height, FLAGS.width,
                       3).astype(np.float32)
  feature_output1 = feature_model(input1, split_features_by_sample=False)
  feature_output2 = feature_model(input2, split_features_by_sample=False)
  flow_output = flow_model(feature_output1, feature_output2, training=False)
  if prev_flow_output is None:
    return flow_output
  for f1, f2 in zip(prev_flow_output, flow_output):
    assert np.max(f1.numpy() - f2.numpy()) < .01


def create_frozen_teacher_models(nccflow):
  """Create a frozen copy of the current nccflow model."""
  nccflow_copy = create_nccflow()
  teacher_feature_model = nccflow_copy.feature_model
  teacher_flow_model = nccflow_copy.flow_model
  # need to create weights in teacher models by calling them
  bogus_input1 = np.random.randn(FLAGS.batch_size, FLAGS.height,
                                 FLAGS.width, 3).astype(np.float32)
  bogus_input2 = np.random.randn(FLAGS.batch_size, FLAGS.height,
                                 FLAGS.width, 3).astype(np.float32)
  existing_model_output = nccflow.feature_model(
      bogus_input1, split_features_by_sample=False)
  _ = teacher_feature_model(bogus_input1, split_features_by_sample=False)
  teacher_feature_model.set_weights(nccflow.feature_model.get_weights())
  teacher_output1 = teacher_feature_model(
      bogus_input1, split_features_by_sample=False)
  teacher_output2 = teacher_feature_model(
      bogus_input2, split_features_by_sample=False)

  # check that both feature models have the same output
  assert np.max(existing_model_output[-1].numpy() -
                teacher_output1[-1].numpy()) < .01
  existing_model_flow = nccflow.flow_model(
      teacher_output1, teacher_output2, training=False)
  _ = teacher_flow_model(teacher_output1, teacher_output2, training=False)
  teacher_flow_model.set_weights(nccflow.flow_model.get_weights())
  teacher_flow = teacher_flow_model(
      teacher_output1, teacher_output2, training=False)
  # check that both flow models have the same output
  assert np.max(existing_model_flow[-1].numpy() -
                teacher_flow[-1].numpy()) < .01
  # Freeze the teacher models.
  for layer in teacher_feature_model.layers:
    layer.trainable = False
  for layer in teacher_flow_model.layers:
    layer.trainable = False

  return teacher_feature_model, teacher_flow_model


def main(unused_argv):
  kitti_EPE_1 = 100.0#kitti-2015
  sintel_EPE_1 = 100.0#sintel-clean
  sintel_EPE_2 = 100.0#sintel-final
  chairs_EPE = 100.0
  num = 5


  if FLAGS.no_tf_function:
    tf.config.experimental_run_functions_eagerly(True)
    print('TFFUNCTION DISABLED')

  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  # Make directories if they do not exist yet.
  if FLAGS.checkpoint_dir and not tf.io.gfile.exists(FLAGS.checkpoint_dir):
    print('Making new checkpoint directory', FLAGS.checkpoint_dir)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
  if FLAGS.checkpoint_dir1 and not tf.io.gfile.exists(FLAGS.checkpoint_dir1):
    print('Making new checkpoint directory', FLAGS.checkpoint_dir1)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir1)
  if FLAGS.checkpoint_dir2 and not tf.io.gfile.exists(FLAGS.checkpoint_dir2):
    print('Making new checkpoint directory', FLAGS.checkpoint_dir2)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir2)
  if FLAGS.plot_dir and not tf.io.gfile.exists(FLAGS.plot_dir):
    print('Making new plot directory', FLAGS.plot_dir)
    tf.io.gfile.makedirs(FLAGS.plot_dir)

  nccflow = create_nccflow()

  if not FLAGS.from_scratch:
    # First restore from init_checkpoint_dir, which is only restored from but
    # not saved to, and then restore from checkpoint_dir if there is already
    # a model there (e.g. if the run was stopped and restarted).
    if FLAGS.init_checkpoint_dir:
      print('Initializing model from checkpoint {}.'.format(
          FLAGS.init_checkpoint_dir))
      nccflow.update_checkpoint_dir(FLAGS.init_checkpoint_dir)
      nccflow.restore(
          reset_optimizer=FLAGS.reset_optimizer,
          reset_global_step=FLAGS.reset_global_step)
      nccflow.update_checkpoint_dir(FLAGS.checkpoint_dir)

    elif FLAGS.checkpoint_dir:
      print('Restoring model from checkpoint {}.'.format(FLAGS.checkpoint_dir))
      nccflow.restore()
  else:
    print('Starting from scratch.')

  print('Making eval datasets and eval functions.')
  if FLAGS.eval_on:
    evaluate, _ = data.make_eval_function(
        FLAGS.eval_on,
        FLAGS.height,
        FLAGS.width,
        progress_bar=True,
        plot_dir=FLAGS.plot_dir,
        num_plots=10)

  if FLAGS.train_on:
    # Build training iterator.
    print('Making training iterator.')
    train_it = data.make_train_iterator(
        FLAGS.train_on,
        FLAGS.height,
        FLAGS.width,
        FLAGS.shuffle_buffer_size,
        FLAGS.batch_size,
        FLAGS.seq_len,
        crop_instead_of_resize=FLAGS.crop_instead_of_resize,
        apply_augmentation=True,
        include_ground_truth=FLAGS.use_supervision,
        resize_gt_flow=FLAGS.resize_gt_flow_supervision,
        include_occlusions=FLAGS.use_gt_occlusions,
    )

    if FLAGS.use_supervision:
      # Since this is the only loss in this setting, and the Adam optimizer
      # is scale invariant, the actual weight here does not matter for now.
      weights = {'supervision': 1.}
    else:
      # Note that self-supervision loss is added during training.
      weights = {
          'photo': FLAGS.weight_photo,
          'ssim': FLAGS.weight_ssim,
          'census': FLAGS.weight_census,
          'smooth1': FLAGS.weight_smooth1,
          'smooth2': FLAGS.weight_smooth2,
          'edge_constant': FLAGS.smoothness_edge_constant,
          'intersect': FLAGS.weight_intersect,
          'flow_blocked': FLAGS.weight_flow_blocked,
      }

      # Switch off loss-terms that have weights < 1e-7.
      weights = {
          k: v for (k, v) in weights.items() if v > 1e-7 or k == 'edge_constant'
      }

    def weight_selfsup_fn():
      step = tf.compat.v1.train.get_or_create_global_step(
      ) % FLAGS.selfsup_step_cycle
      # Start self-supervision only after a certain number of steps.
      # Linearly increase self-supervision weight for a number of steps.
      ramp_up_factor = tf.clip_by_value(
          float(step - (FLAGS.selfsup_after_num_steps - 1)) /
          float(max(FLAGS.selfsup_ramp_up_steps, 1)), 0., 1.)
      return FLAGS.weight_selfsup * ramp_up_factor

    distance_metrics = {
        'photo': FLAGS.distance_photo,
        'census': FLAGS.distance_census,
    }

    print('Starting training loop.')
    log = dict()
    epoch = 0

    teacher_feature_model = None
    teacher_flow_model = None
    test_frozen_flow = None

    while True:
      current_step = tf.compat.v1.train.get_or_create_global_step().numpy()

      # Set which occlusion estimation methods could be active at this point.
      # (They will only be used if occlusion_estimation is set accordingly.)
      occ_active = {
          'uflow':
              FLAGS.occlusion_estimation == 'uflow',
          'brox':
              current_step > FLAGS.occ_after_num_steps_brox,
          'wang':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wang4':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wangthres':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wang4thres':
              current_step > FLAGS.occ_after_num_steps_wang,
          'fb_abs':
              current_step > FLAGS.occ_after_num_steps_fb_abs,
          'forward_collision':
              current_step > FLAGS.occ_after_num_steps_forward_collision,
          'backward_zero':
              current_step > FLAGS.occ_after_num_steps_backward_zero,
      }

      current_weights = {k: v for k, v in weights.items()}

      # Prepare self-supervision if it will be used in the next epoch.
      if FLAGS.weight_selfsup > 1e-7 and (
          current_step % FLAGS.selfsup_step_cycle
      ) + FLAGS.epoch_length > FLAGS.selfsup_after_num_steps:

        # Add selfsup weight with a ramp-up schedule. This will cause a
        # recompilation of the training graph defined in nccflow.train(...).
        current_weights['selfsup'] = weight_selfsup_fn

        # Freeze model for teacher distillation.
        if teacher_feature_model is None and FLAGS.frozen_teacher:
          # Create a copy of the existing models and freeze them as a teacher.
          # Tell nccflow about the new, frozen teacher model.
          teacher_feature_model, teacher_flow_model = create_frozen_teacher_models(
              nccflow)
          nccflow.set_teacher_models(
              teacher_feature_model=teacher_feature_model,
              teacher_flow_model=teacher_flow_model)
          test_frozen_flow = check_model_frozen(
              teacher_feature_model, teacher_flow_model, prev_flow_output=None)

          # Check that the model actually is frozen.
          if FLAGS.frozen_teacher and test_frozen_flow is not None:
            check_model_frozen(
                teacher_feature_model,
                teacher_flow_model,
                prev_flow_output=test_frozen_flow)

      # Train for an epoch and save the results.
      log_update = nccflow.train(
          train_it,
          weights=current_weights,
          num_steps=FLAGS.epoch_length,
          progress_bar=True,
          plot_dir=FLAGS.plot_dir if FLAGS.plot_debug_info else None,
          distance_metrics=distance_metrics,
          occ_active=occ_active)


      for key in log_update:
        if key in log:
          log[key].append(log_update[key])
        else:
          log[key] = [log_update[key]]

      if FLAGS.checkpoint_dir and not FLAGS.no_checkpointing:
        nccflow.save()

      # Print losses from last epoch.
      plotting.print_log(log, epoch)



      if FLAGS.eval_on and FLAGS.evaluate_during_train and epoch % num == 0:
        # Evaluate
        #if FLAGS.eval_on and FLAGS.evaluate_during_train:
        eval_results = evaluate(nccflow)
        plotting.print_eval(eval_results)
        status = ''.join(
            ['{}: {:.6f}, '.format(key, eval_results[key]) for key in sorted(eval_results)])
        eval_on = FLAGS.eval_on
        for format_and_path in eval_on.split(';'):
            data_format, path = format_and_path.split(':')

        if 'kitti' in data_format:
            EPE_1 = float(status.split(',')[1].split(':')[1][1:])
            if EPE_1 < kitti_EPE_1:#kitti-2015
                nccflow.save_1()
                kitti_EPE_1 = EPE_1
        elif 'sintel' in data_format:
            EPE_1 = float(status.split(',')[0].split(':')[1][1:])
            EPE_2 = float(status.split(',')[6].split(':')[1][1:])
            if EPE_1 < sintel_EPE_1:  # sintel-clean
                nccflow.save_1()
                sintel_EPE_1 = EPE_1
            if EPE_2 < sintel_EPE_2:  # sintel-fianl
                nccflow.save_2()
                sintel_EPE_2 = EPE_2
        elif 'chairs' in data_format:
          EPE = float(status[12:20])
          if EPE < chairs_EPE:
            nccflow.save_1()
            chairs_EPE = EPE

      if current_step >= FLAGS.num_train_steps:
        break

      epoch += 1

  else:
    print('Specify flag train_on to enable training to <format>:<path>;... .')
    print('Just doing evaluation now.')
    eval_results = evaluate(nccflow)
    if eval_results:
      plotting.print_eval(eval_results)
    print('Evaluation complete.')


if __name__ == '__main__':
  app.run(main)
