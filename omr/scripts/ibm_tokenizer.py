
import re, sys
from tqdm import tqdm

token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

if __name__ == '__main__':
    #smiles_path = sys.argv[1]
    #tok_path = sys.argv[2]
    smiles_path = '/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/data/smiles_tokens.txt'
    tok_path = '/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/data/smiles_vocab.txt'
    with open(tok_path, "w") as fw:
        for mol in tqdm(open(smiles_path, "r")):
            mol = mol.strip()
            toks = re.findall(token_regex, mol)
            print >> fw, " ".join(toks)

