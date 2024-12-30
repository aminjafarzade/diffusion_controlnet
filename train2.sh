export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/controlnet_dogs"   # output directory of each run

accelerate launch train.py \
--seed=0 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=JFoz/dog-poses-controlnet-dataset \
--resolution=512 \
--learning_rate=1e-5 \
--num_train_epochs=5 \
--validation_image "./data_2/conditioning_image_1.png" "./data_2/conditioning_image_2.png" \
--validation_prompt "a small yorkshire terrier dog is sitting on a black background" "a brown dog walking through the grass" \
--train_batch_size=1 \
--image_column="original_image" \
--conditioning_image_column="conditioning_image" \
--caption_column="caption" \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard" \
