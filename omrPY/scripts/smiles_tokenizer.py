# from https://github.com/tuffery/Frog2
# major changes: removed Handler class dependency
#                added aromatic selenium

import re, string

# To verify this is correct, run
#    support.make_re_pattern(support.element_symbols + support.aromatic_symbols)
# Yes, I included aromatics in this list.
element_symbols_pattern = \
    r"C[laroudsemf]?|Os?|N[eaibdpos]?|S[icernbmg]?|P[drmtboau]?|"    \
    r"H[eofgas]?|c|n|o|s[e]?|p|A[lrsgutcm]|B[eraik]?|Dy|E[urs]|F[erm]?|"    \
    r"G[aed]|I[nr]?|Kr?|L[iaur]|M[gnodt]|R[buhenaf]|T[icebmalh]|" \
    r"U|V|W|Xe|Yb?|Z[nr]|\*"

atom_fields = [
        "raw_atom",
        "open_bracket",
        "weight",
        "element",
        "chiral_count",
        "chiral_named",
        "chiral_symbols",
        "hcount",
        "positive_count",
        "positive_symbols",
        "negative_count",
        "negative_symbols",
        "error_1",
        "error_2",
        "close_bracket",
        "error_3",
        ]

atom = re.compile(r"""
(?P<raw_atom>Cl|Br|[cnospBCNOFPSI]) |    # "raw" means outside of brackets
(
    (?P<open_bracket>\[)                                 # Start bracket
    (?P<weight>\d+)?                                         # Atomic weight (optional)
    (                                                                        # valid term or error
     (                                                                     #     valid term
        (?P<element>""" + element_symbols_pattern + r""")    # element or aromatic
        (                                                                    # Chirality can be
         (?P<chiral_count>@\d+) |                    #     @1 @2 @3 ...
         (?P<chiral_named>                                 # or
             @TH[12] |                                             #     @TA1 @TA2
             @AL[12] |                                             #     @AL1 @AL2
             @SP[123] |                                            #     @SP1 @SP2 @SP3
             @TB(1[0-9]?|20?|[3-9]) |                #     @TB{1-20}
             @OH(1[0-9]?|2[0-9]?|30?|[4-9])) | # @OH{1-30}
         (?P<chiral_symbols>@+)                        # or @@@@@@@...
        )?                                                                 # and chirality is optional
        (?P<hcount>H\d*)?                                    # Optional hydrogen count
        (                                                                    # Charges can be
         (?P<positive_count>\+\d+) |             #     +<number>
         (?P<positive_symbols>\++) |             #     +++...    This includes the single '+'
         (?P<negative_count>-\d+)    |             #     -<number>
         (?P<negative_symbols>-+)                    #     ---...    including a single '-'
        )?                                                                 # and are optional
        (?P<error_1>[^\]]+)?                             # If there's anything left, it's an error
    ) | (                                                                # End of parsing stuff in []s, except
        (?P<error_2>[^\]]*)                                # If there was an error, we get here
    ))
    ((?P<close_bracket>\])|                            # End bracket
     (?P<error_3>$))                                         # unexpectedly reached end of string
)
""", re.X)

bond_fields = ["bond"]
bond = re.compile(r"(?P<bond>[=#/\\:~-])")

dot_fields = ["dot"]
dot = re.compile(r"(?P<dot>\.)")

closure_fields = ["closure"]
closure = re.compile(r"(?P<closure>\d|%\d\d?)")

close_branch_fields = ["close_branch"]
close_branch = re.compile(r"(?P<close_branch>\))")

open_branch_fields = ["open_branch"]
open_branch = re.compile(r"(?P<open_branch>\()")

# Make from the state name to
#     1. the regular expession to try to match and
#     2. the list of field names in that regexp.
# (There's no way to use sre to get #2 as an ordered list so
#    it needs to be done manually.)
info = {
        "atom": (atom, atom_fields),
        "bond": (bond, bond_fields),
        "dot": (dot, dot_fields),
        "closure": (closure, closure_fields),
        "close_branch": (close_branch, close_branch_fields),
        "open_branch": (open_branch, open_branch_fields),
        }

# Mapping from current state to allowed states
table = {
        # Could allow a dot
        "start": ("atom",),

        # CC, C=C, C(C)C, C(C)C, C.C, C1CCC1
        "atom": ("atom", "bond", "close_branch", "open_branch", "dot", "closure"),

        # C=C, C=1CCC=1
        "bond": ("atom", "closure"),

        # C(C)C, C(C)=C, C(C).C, C(C(C))C, C(C)(C)C
        "close_branch": ("atom", "bond", "dot", "close_branch", "open_branch"),

        # C(C), C(=C), C(.C) (really!)
        "open_branch": ("atom", "bond", "dot"),

        # C.C    -- allow a dot? as in C..C
        "dot": ("atom",),

        # C1CCC1, C1=CCC1, C12CC1C2, C1C(CC)1, C1(CC)CC1, c1ccccc1.[NH4+]
        "closure": ("atom", "bond", "closure", "close_branch", "open_branch", "dot"),
}

def raise_error(txt, pos, remaining):
    raise ValueError('{} at position {}, starting w/ {}'.format(txt, pos, remaining))

# Parse a SMILES string and print the events found
def tokenize(s, include_start_end=False):
        split_smiles = []
        add_token = split_smiles.append

        expected = table["start"]
        if include_start_end:
            add_token(('start_smiles', 0, '<START>'))
        n = len(s)
        i = 0

        while i < n:
                # Of the expected states, find one that matches the
                # text at the current position.
                for state in expected:
                        pat, fields = info[state]
                        m = pat.match(s, i)
                        if m:
                                break
                else:
                        # No matches found, so this was an error
                        raise_error("Unknown character", i, s[i:])
                        # The handler is allowed to not throw an
                        # exception, but we are done, so return.
                        return
                #print "New state:", state,

                # Get the dictionary of matched name groups
                d = m.groupdict()

                # Go through the list of fields that could have matched.
                # Needs to go in a order so the token text can be converted
                # back into the original string.
                for field in fields:
                        # See if there was a match for the given named field
                        if d[field] is not None:
                                # Was it an error match?
                                if field[:5] == "error":
                                        pos = m.start(field)
                                        if field == "error_3":
                                                raise_error("Missing ']'", pos, s[pos:])
                                        else:
                                                raise_error("Unknown character", pos, s[pos:])
                                        return
                                # Success, so send the token to the callback
                                #print "--> ", m.group(field), field
                                add_token((field, i, m.group(field)))

                # Get the new list of expected states, and move to
                # the end of the previous match.
                expected = table[state]
                i = m.end(0)

        if include_start_end:
            add_token(('end_smiles', 0, '<END>'))
        return split_smiles 


def tokenize_all(data, outfile, explicit=False):
    '''Get the vocabulary from a SMILES file'''
    from collections import defaultdict
    vocab_counts = defaultdict(int)

    from tqdm import tqdm
    num_tokens = 0
    N = 0
    fw = open(outfile, "w")
    for smiles in tqdm(open(data)):
        # print(smiles)
        smiles = smiles.strip()
        try:
            if explicit:
                mol = Chem.MolFromSmiles(smiles)
                Chem.Kekulize(mol)
                smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                #     isomericSmiles=True,
                #     allBondsExplicit=True)
                    # allHsExplicit=True)

            tokens = tokenize(smiles)
            print >> fw, " ".join(e[2] for e in tokens)
            num_tokens += len(tokens)
            for (f, i, t) in tokens:
                vocab_counts[t] += 1
            N += 1

        except ValueError as e:
            print(e)
            print(smiles)

    print('Average product length: {} tokens'.format(num_tokens / float(N)))
    return vocab_counts


if __name__ == '__main__':

    import rdkit.Chem as Chem 
    import sys

    smiles_path = sys.argv[1]
    tok_path = sys.argv[2]
    print('Getting vocab from training set only, using explicit tokens')
    vocab_counts = tokenize_all(smiles_path, tok_path)

    print(vocab_counts)
    print('{} unique'.format(len(vocab_counts)))

