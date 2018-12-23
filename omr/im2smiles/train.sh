
BASE_DIR=../data/jin_uspto
LOG_FILE=./train.100k.log
if [ -f $LOG_FILE ]; then
    rm $LOG_FILE
fi

th src/train.lua -phase train -gpu_id 3 \
    -model_dir model \
    -input_feed -prealloc \
    -data_base_dir $BASE_DIR/ \
    -data_path $BASE_DIR/train_filter.lst \
    -val_data_path $BASE_DIR/valid_filter.lst \
    -label_path $BASE_DIR/smiles.120k \
    -vocab_file $BASE_DIR/smiles_120k_vocab.txt \
    -max_num_tokens 150 -max_image_width 500 -max_image_height 160 \
    -target_embedding_size 50 -encoder_num_hidden 128 \
    -batch_size 20 -beam_size 1 -num_epochs 20 -log_path $LOG_FILE
