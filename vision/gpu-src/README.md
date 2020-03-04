# GPU CIFAR10 Models

These are models provided alongside Tensorflow, from https://github.com/tensorflow/models/tree/v1.13.0/official/resnet.

## Setup

```
python3 official/resnet/cifar10_download_and_extract.py
```

## Initial training

```
BASE_NETWORK="resnet" # or vgg
SIZE="20" # or 56,110, or 16_nofc or VGG

function train() {
    MODEL_DIR="${1}"; shift

    python "official/${BASE_NETWORK}/cifar10_main.py" \
       "--${BASE_NETWORK}_size" "${SIZE}" \
       --batch_size 128 \
       --datasets_num_parallel_batches 1 \
       --model_dir "${MODEL_DIR}" \
       --lottery_results_dir "${MODEL_DIR}" \
       --lottery_checkpoint_iters 0,4701,12514,20326,28139,35951,43764,51576,59389,67201,71108 \
       "${@}"
}

train ./train_model_results --max_train_steps 71108
```

## Rewinding weights and learning rate

```
REWIND_ITERATION=4701
PREV_DIR=train_model_results
RESULT_DIR=rewind_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
mkdir -p ${RESULT_DIR}
cp "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}"* "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
echo 'model_checkpoint_path: "'"checkpoint_iter_${REWIND_ITERATION}"'"' > "${RESULT_DIR}/checkpoint"

# prune to 80% density, rewind to iteration 4701, and re-train.
train ${RESULT_DIR} --max_train_steps 71108 --lottery_pruning_method prune_all_to_global_80.0 --lottery_reset_to ${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION} --lottery_prune_at ${PREV_DIR}/checkpoint_iter_final
```

## Rewinding learning rate
```
REWIND_ITERATION=4701
PREV_DIR=train_model_results
RESULT_DIR=rewind_lr_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
mkdir -p ${RESULT_DIR}
cp "${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}"* "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
echo 'model_checkpoint_path: "'"checkpoint_iter_${REWIND_ITERATION}"'"' > "${RESULT_DIR}/checkpoint"

# prune to 80% density, rewind global step (i.e., LR) to iteration 4701, and re-train.
train ${RESULT_DIR} --max_train_steps 71108 --lottery_pruning_method prune_all_to_global_80.0 --lottery_reset_to ${PREV_DIR}/checkpoint_iter_final --lottery_prune_at ${PREV_DIR}/checkpoint_iter_final --lottery_reset_global_step_to ${PREV_DIR}/checkpoint_iter_${REWIND_ITERATION}
```

## Fine-tuning
```
REWIND_ITERATION=4701
FINETUNE_EPOCHS=$(expr \( 71108 - ${REWIND_ITERATION} \) / \( 71108 / 182 \) )
MAX_TRAIN_STEPS=$(expr 71108 + 71108 - ${REWIND_ITERATION})
PREV_DIR=train_model_results
RESULT_DIR=finetune_model_results

# tensorflow can be annoying if it doesn't think it's starting from the right checkpoint
mkdir -p ${RESULT_DIR}
cp "${PREV_DIR}/checkpoint_iter_final"* "${PREV_DIR}/graph.pbtxt" "${RESULT_DIR}/"
echo 'model_checkpoint_path: "'"checkpoint_iter_final"'"' > "${RESULT_DIR}/checkpoint"

# prune to 80% density, train. lottery_force_learning_rate is explicitly set to the LR that the model should use while fine-tuning
train ${RESULT_DIR} --max_train_steps ${MAX_TRAIN_STEPS} --lottery_pruning_method prune_all_to_global_80.0 --lottery_reset_to ${PREV_DIR}/checkpoint_iter_final --lottery_prune_at ${PREV_DIR}/checkpoint_iter_final --lottery_force_learning_rate "0.001"
```

## Checking results

This awful script will print out the (training step, test accuracy) values recorded during training.
```
for f in train_model_results/eval/events.out.*; do
    TF_CPP_MIN_LOG_LEVEL=3 python3 -c 'import tensorflow as tf; tf.enable_eager_execution(); tf.logging.set_verbosity(tf.logging.ERROR); print("\n".join(map(str, list((e.step, v.simple_value) for e in map(lambda x: tf.Event.FromString(x.numpy()), tf.data.TFRecordDataset("'"${f}"'")) for v in e.summary.value if v.tag == "accuracy"))))';
done
```
