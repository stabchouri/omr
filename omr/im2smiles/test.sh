
BASE_DIR=../data/jin_uspto
LOG_FILE=./valid.100k.log
if [ -f $LOG_FILE ]; then
    rm $LOG_FILE
fi

th src/train.lua -phase test -gpu_id 2 -load_model -model_dir model -visualize \
    -data_base_dir $BASE_DIR \
    -data_path $BASE_DIR/valid_filter.lst \
    -label_path $BASE_DIR/smiles.120k \
    -output_dir valid_results \
    -max_num_tokens 150 -max_image_width 500 -max_image_height 160 \
    -batch_size 5 -beam_size 5 -log_path $LOG_FILE
