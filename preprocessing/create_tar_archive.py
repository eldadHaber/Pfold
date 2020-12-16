import webdataset as wds

fastafile = './../Data/Dummy_file.fasta'
# sink = wds.TarWriter("dest.tar")
sink = wds.ShardWriter("dest-%09d.tar", maxcount=100000000)

idx = 0
with open(fastafile, 'r') as f:
    while True:
        idx += 1
        line = f.readline()

        if not line:
            break
        sink.write({
            "__key__": "sample%09d" % idx,
            "seq.pyd": line,
        })
sink.close()
print("done")
#
#
# dataset_path = "/imagenet/imagenet-train-{000000..001281}.tar"
#
# dataset = (
#     wds.Dataset(dataset_path)
#     .shuffle(100)
# )
#
# loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8)
