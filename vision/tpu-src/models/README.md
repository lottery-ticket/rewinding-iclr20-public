# ImageNet TPU models

These are models provided alongside Tensorflow, from  https://github.com/tensorflow/tpu/tree/98497e0b/models/official/resnet.

## Setup

Download then follow the instructions to format the ImageNet dataset [here](https://github.com/tensorflow/tpu/tree/98497e0b/models/official/resnet).

Start a TPU v2-8 or v3-8, using Tensorflow 1.13.
Set the variable `TPU_NAME` to refer to the name of this TPU.
Set up a gcloud bucket that you can write to and read from within the Docker instance (take a look at `vision/docker/posthoc_setup.sh` with a [private key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) to see how we do this).

## Initial training

```
DATA_DIR=WHEREVER_IMAGENET_IS_LOCATED
RESNET_DEPTH=50 # or 34

function train() {
    MODEL_DIR="${1}"; shift

    python "official/resnet/resnet_main.py" \
       --hparams_file official/resnet/configs/cloud/v3-8.yaml \
       --data_dir "${DATA_DIR}" \
       --resnet_depth ${RESNET_DEPTH} \
       --model_dir "${MODEL_DIR}" \
       --lottery_results_dir "${MODEL_DIR}" \
       --tpu "${TPU_NAME}" \
       --lottery_checkpoint_iters 0,11259,22518,33777,45036,56295,67554,78813,90072,101331,112590 \
       "${@}"
}

# TPUs can only read from / write to gcloud directories, so make and use one of those
train gs://${YOUR_BUCKET_HERE}/train_model_results
```

## Rewinding weights and learning rate

```
REWIND_ITERATION=11259
PREV_DIR=gs://${YOUR_BUCKET_HERE}/train_model_results
RESULT_DIR=gs://${YOUR_BUCKET_HERE}/rewind_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
gsutil cp "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}"'*' "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
# also it's hard to write a file to gcloud
tmpfile="$(mktemp)"
echo 'model_checkpoint_path: "'"checkpoint_iter_${REWIND_ITERATION}"'"' > "${tmpfile}"
gsutil cp "${tmpfile}" "${RESULT_DIR}/checkpoint"
rm "${tmpfile}"

# prune to 80% density, rewind to iteration 11259, and re-train.
train ${RESULT_DIR} \
    --train_steps 112590 \
    --lottery_pruning_method "prune_all_to_global_80.0" \
    --lottery_reset_to "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}" \
    --lottery_prune_at "${PREV_DIR}/checkpoint_iter_final"
```

## Rewinding learning rate
```
REWIND_ITERATION=11259
PREV_DIR=gs://${YOUR_BUCKET_HERE}/train_model_results
RESULT_DIR=gs://${YOUR_BUCKET_HERE}/rewind_lr_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
gsutil cp "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}"'*' "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
# also it's hard to write a file to gcloud
tmpfile="$(mktemp)"
echo 'model_checkpoint_path: "'"checkpoint_iter_${REWIND_ITERATION}"'"' > "${tmpfile}"
gsutil cp "${tmpfile}" "${RESULT_DIR}/checkpoint"
rm "${tmpfile}"

# prune to 80% density, rewind to iteration 11259, and re-train.
train ${RESULT_DIR} \
    --train_steps 112590 \
    --lottery_pruning_method "prune_all_to_global_80.0" \
    --lottery_reset_to "${PREV_DIR}/checkpoint_iter_final" \
    --lottery_reset_global_step_to "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}" \
    --lottery_prune_at "${PREV_DIR}/checkpoint_iter_final"
```

## Fine-tuning
```
REWIND_ITERATION=11259
MAX_TRAIN_STEPS=$(expr 112590 + 112590 - ${REWIND_ITERATION})
PREV_DIR=gs://${YOUR_BUCKET_HERE}/train_model_results
RESULT_DIR=gs://${YOUR_BUCKET_HERE}/rewind_lr_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
gsutil cp "${PREV_DIR}/checkpoint_iter_final"'*' "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
# also it's hard to write a file to gcloud
tmpfile="$(mktemp)"
echo 'model_checkpoint_path: "'"checkpoint_iter_final"'"' > "${tmpfile}"
gsutil cp "${tmpfile}" "${RESULT_DIR}/checkpoint"
rm "${tmpfile}"

# prune to 80% density, fine-tune
# lottery_force_learning_rate is explicitly set to the LR that the model should use while fine-tuning
train ${RESULT_DIR} \
    --train_steps ${MAX_TRAIN_STEPS} \
    --lottery_pruning_method "prune_all_to_global_80.0" \
    --lottery_force_learning_rate "0.0004" \
    --lottery_reset_to "${PREV_DIR}/checkpoint_iter_final" \
    --lottery_prune_at "${PREV_DIR}/checkpoint_iter_final"
```

## Checking results

This awful script will print out the (training step, test accuracy) values recorded during training.
```
for f in `gsutil ls gs://${YOUR_BUCKET_HERE}/train_model_results/eval/events.out.'*'`; do
    TF_CPP_MIN_LOG_LEVEL=3 python3 -c 'import tensorflow as tf; tf.enable_eager_execution(); tf.logging.set_verbosity(tf.logging.ERROR); print("\n".join(map(str, list((e.step, v.simple_value) for e in map(lambda x: tf.Event.FromString(x.numpy()), tf.data.TFRecordDataset("'"${f}"'")) for v in e.summary.value if v.tag == "top_1_accuracy"))))';
done
```
