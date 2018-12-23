import rdkit
import rdkit.Chem as Chem
import sys

for l in open(sys.argv[1]):
    l = l.strip()
    mol = Chem.MolFromSmiles(l)
    Chem.Kekulize(mol)
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    print smiles
