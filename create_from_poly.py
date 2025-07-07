from search_utils import ConfigurationSearcher
from converter import generate_mace_fingerprints
from configuration import Configuration

type_map = {0: 6, 1: 1, 2: 8}

id_to_name = {
    'ID-00001': 'LAC5',
    'ID-00003': 'LAC9',
    'ID-00005': 'LAC6',
    'ID-00007': 'LAC2',
    'ID-00009': 'LAC7',
    'ID-00011': 'LAC8',
    'ID-00015': 'LAC3',
    'ID-00049': 'CK1',
    'ID-00050': 'ETH',
    'ID-00051': 'CK2',
    'ID-00052': 'LAC1',
    'ID-00053': 'CK3',
    'ID-00993': 'CK4',
    'ID-00994': 'CK5',
    'ID-00995': 'CK6',
    'ID-01050': 'OTH1',
    'ID-01052': 'OTH3',
    'ID-01053': 'OTH4',
    'ID-01054': 'OTH5',
    'ID-01055': 'OTH6',
    'ID-01056': 'OTH7',
    'ID-01058': 'OTH8',
    'ID-01060': 'OTH2'
}

confs = []
for cflist in id_to_name.keys():
    #print(cflist)
    added = Configuration.from_file(f"/home/norekhov/_Alekseev/bmstu_polymers/{cflist}/output.cfg")
    print(cflist, len(added))
    confs = added
    fingerprints = generate_mace_fingerprints(added, "2023-12-03-mace-mp.model", type_map, device='cuda')
    np.save(cflist, fingerprints)
