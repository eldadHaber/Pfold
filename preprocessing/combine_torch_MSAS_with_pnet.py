#This routine assumes you have already converted MSAS into torch format, using process_raw_MSAS. Not that the MSAS should already have been sanitized (Check for duplicates and remove any MSAS wit insufficient information ect.)
#Furthermore it assumes you have proteinnet data converted into torch format as well, which can be done using dataloader_pnet


if __name__ == '__main__':
    pnet_folder = ''
    msa_folder = ''
    output_folder = ''

    #This code will read in all the data in pnet and store it in memory.
    #It will then go through the data in the MSA folder one by one and pair try to pair it up against pnet data, when a match is found the data will be save to output combined, and the data will be removed from pnet, such that the search space gets smaller and smaller as it progresses






