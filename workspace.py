from pathlib import Path

# 0) Where am I?
print("CWD:", Path.cwd())

# 1) Point to the parent dir (the one you said returns True/True)
parent = Path(r"results/diff_clients_nets/rnd/data_setCIFAR100_algCOMET_clustersNone_serverALEXNET_clients25_clientrndNet_alpha5_ratio10")
print(parent.exists(), parent.is_dir())

# 2) List what's actually there (helps catch typos/whitespace/case/extension)
for x in parent.iterdir():
    print(" -", x.name)

# 3) Show only JSONs we can actually see (case-insensitive)
print("JSONs seen:")
for x in parent.glob("*.json"):
    print(" *", x.name)
for x in parent.glob("*.JSON"):
    print(" *", x.name)

# 4) Try your exact file, but ABSOLUTE
p = (Path.cwd() / Path(
    r"results/diff_clients_nets/rnd/data_setCIFAR100_algCOMET_clustersNone_serverALEXNET_clients25_clientrndNet_alpha5_ratio10/data_setCIFAR100_algCOMET_clustersNone_serverALEXNET_clients25_clientrndNet_alpha5_ratio10_seed1.json"
)).resolve()
print("ABS exists/is_file:", p.exists(), p.is_file())
print("Length:", len(str(p)))
