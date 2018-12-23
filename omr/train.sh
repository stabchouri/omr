
th im2markup/src/train.lua -phase train -gpu_id 1 \
    -model_dir model \
    -input_feed -prealloc \
    -data_base_dir data/jin_uspto \
    -data_path data/jin_uspto/train_filter.lst \
    -val_data_path data/jin_uspto/validate_filter.lst \
    -label_path data/jin_uspto/smiles.120k \
    -vocab_file data/jin_uspto/smiles_120k_vocab.txt \
    -max_num_tokens 150 -max_image_width 500 -max_image_height 160 \
    -batch_size 20 -beam_size 1
