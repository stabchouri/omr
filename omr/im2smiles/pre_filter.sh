
# filter large images
# python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/smi2png_processed/ --label-path data/smiles.txt --data-path data/train.lst --output-path data/train_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../data/processed/png_processed --label-path ../data/out1.txt --data-path ../data/validate.lst --output-path ../data/validate_filter.lst
#python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/smi2png_processed/ --label-path data/smiles.txt --data-path data/test.lst --output-path data/test_filter.lst


# generate vocab
#python scripts/preprocessing/generate_latex_vocab.py --data-path data/train_filter.lst --label-path data/smiles.txt --output-file data/smiles_vocab.txt
