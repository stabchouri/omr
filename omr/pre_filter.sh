
# filter large images
BASE_DIR=data/jin_uspto/
SCPT_DIR=im2markup/scripts/preprocessing/

### preprocessing images with specific downsampling ratio (crop, group, downsample)
dsr=2
python im2markup/scripts/preprocessing/preprocess_images.py --input-dir data/jin_uspto/test.img/png_10k --output-dir data/jin_uspto/test.img/png_10k_processed_dsr${dsr} --downsample-ratio ${dsr} --num-threads 10

# python $SCPT_DIR/preprocess_filter.py --filter --image-dir $BASE_DIR --label-path $BASE_DIR/smiles.120k --data-path $BASE_DIR/train.lst --output-path $BASE_DIR/train_filter.lst
# python $SCPT_DIR/preprocess_filter.py --filter --image-dir $BASE_DIR --label-path $BASE_DIR/smiles.120k --data-path $BASE_DIR/valid.lst --output-path $BASE_DIR/valid_filter.lst
# python $SCPT_DIR/preprocess_filter.py --filter --image-dir $BASE_DIR --label-path $BASE_DIR/smiles.120k --data-path $BASE_DIR/test.lst --output-path $BASE_DIR/test_filter.lst

python $SCPT_DIR/preprocess_filter.py --filter --image-dir $BASE_DIR --label-path $BASE_DIR/smiles.120k --data-path $BASE_DIR/test_dsr2.lst --output-path $BASE_DIR/test_dsr2_filter.lst


# generate vocab
# python $SCPT_DIR/generate_latex_vocab.py --data-path $BASE_DIR/train_filter.lst --label-path $BASE_DIR/smiles.120k --output-file $BASE_DIR/smiles_120k_vocab.txt
