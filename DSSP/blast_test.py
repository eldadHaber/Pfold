from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

# seq = 'DIVLTQSPDITAASLGQKVTITCSASSSVSYMHWYQQKSGTSPKPWIFEISKLASGVPARFSGSGSGTSYSLTISSMEAEDAAIYYCQQWNYPFTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC'
seq = 'DIVLTQSPDITAASLGQKVTITCSASSSVSYMHWYQQKSGTSPKPWIFEI'
result_handle = NCBIWWW.qblast("blastp", "pdb", seq)
blast_record = NCBIXML.read(result_handle)

from Bio.Blast import NCBIXML
blast_records = NCBIXML.parse(result_handle)


match_found = False
for alignment in blast_record.alignments:
    for hsp in alignment.hsps:
        if hsp.query == seq and hsp.match == seq and hsp.sbjct == seq:
            match_found = True
            pbd_id = alignment.accession.split('_')[0]
            chain_id = alignment.accession.split('_')[1]
            print(alignment.accession)
            break
    if match_found:
        break

# from Bio.Blast.Applications import NcbiblastpCommandline
# blastx_cline = NcbiblastpCommandline(query="seq2.fasta", db="pdb_seqres.txt", evalue=0.001, outfmt=5, out="/home/tue/PycharmProjects/dssp/seq2.xml")
#
# print(blastx_cline)
# # blastp -query seq2.seq -db pdb_seqres.txt -out seq2.txt -outfmt 7
#
#
#

print("done")