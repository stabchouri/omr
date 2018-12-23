import rdkit
import rdkit.Chem as Chem
import sys

def remove_atommap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)
    # return Chem.MolToSmiles(mol)

rxn_file = sys.argv[1]
mol_file = sys.argv[2]

molecules = set()
with open(mol_file, "w") as fw:
    for i, reaction in enumerate(open(rxn_file, "r")):
        if i % 1000 == 0:
            print >> sys.stderr, i

        reaction = reaction.strip().split(" ")[0]
        reactants, products = reaction.split(">>")
        reactants = reactants.split(".")
        # a product may contain more than one molecules
        products = products.split(".")
        for mol in reactants:
            unmapped_mol = remove_atommap(mol)
            # if unmapped_mol == "CC1=C(C(C2=C(C)C3=CC=CC=C3N2)C2=CC=C([N](=O)=O)C=C2)NC2=CC=CC=C21":
            # if unmapped_mol == "C=CC[N]1=CC=CC=C1":
            #     print >> sys.stderr, i+1
            if unmapped_mol is not None:
                molecules.add(unmapped_mol)
        for mol in products:
            unmapped_mol = remove_atommap(mol)
            # if unmapped_mol == "CC1=C(C(C2=C(C)C3=CC=CC=C3N2)C2=CC=C([N](=O)=O)C=C2)NC2=CC=CC=C21":
            # if unmapped_mol == "C=CC[N]1=CC=CC=C1":
            #     print >> sys.stderr, i+1
            if unmapped_mol is not None:
                molecules.add(unmapped_mol)

    for mol in molecules:
        print >> fw, mol

