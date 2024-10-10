import numpy as np

synrs3d_datasetname = set(['terrain_g05_mid_v1', 'grid_g05_mid_v2', 'terrain_g05_low_v1', 'terrain_g05_high_v1', 'terrain_g005_mid_v1',
                           'terrain_g005_low_v1', 'grid_g005_mid_v2', 'terrain_g005_high_v1', 'terrain_g1_mid_v1', 'terrain_g1_low_v1',
                           'terrain_g1_high_v1', 'grid_g005_mid_v1', 'grid_g005_low_v1', 'grid_g005_high_v1', 'grid_g05_mid_v1', 
                           'grid_g05_low_v1', 'grid_g05_high_v1'])
ISPRS_datasetname = set(['vaihingen', 'potsdam'])

real_datasetname = set(['DFC18', 'DFC19_JAX', 'DFC19_OMA', 'DFC23', 'geonrw_urban', 'geonrw_rural', 'vaihingen', 'potsdam', 'nagoya', 'tokyo', 'OGC_JAX', 'OGC_ARG', 'OGC_ATL', 'OGC_OMA'])

ss_datasetname = set(['SParis_03', 'SParis_05', 'SVenice_03', 'SVenice_05', 
                      'terrain_g05_mid_v1', 'grid_g05_mid_v2', 'terrain_g05_low_v1', 'terrain_g05_high_v1', 'terrain_g005_mid_v1',
                      'terrain_g005_low_v1', 'grid_g005_mid_v2', 'terrain_g005_high_v1', 'terrain_g1_mid_v1', 'terrain_g1_low_v1',
                      'terrain_g1_high_v1', 'grid_g005_mid_v1', 'grid_g005_low_v1', 'grid_g005_high_v1', 'grid_g05_mid_v1', 
                      'grid_g05_low_v1', 'grid_g05_high_v1',
                      'DFC19_JAX', 'DFC19_OMA', 'vaihingen', 'potsdam', 'OEM'])
syn_ss_datasetname = set(['terrain_g05_mid_v1', 'grid_g05_mid_v2', 'terrain_g05_low_v1', 'terrain_g05_high_v1', 'terrain_g005_mid_v1',
                          'terrain_g005_low_v1', 'grid_g005_mid_v2', 'terrain_g005_high_v1', 'terrain_g1_mid_v1', 'terrain_g1_low_v1',
                          'terrain_g1_high_v1', 'grid_g005_mid_v1', 'grid_g005_low_v1', 'grid_g005_high_v1', 'grid_g05_mid_v1', 
                          'grid_g05_low_v1', 'grid_g05_high_v1'])
real_ss_datasetname = set(['DFC19_JAX', 'DFC19_OMA', 'vaihingen', 'potsdam'])

OEM_datasetname = set(['OEM'])

dataset_num_classes = {'DFC19': 5, 'ISPRS': 6, 'OEM': 8}

combination_relabel_rules = {
    "OEM": {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 2},
    "DFC19": {2: 0, 5: 1, 6: 2, 9: 0, 17: 0, 65: 2},
    "ISPRS": {0: 0, 1: 2, 2: 0, 3: 1, 4: 0, 5: 0},
    "SYNTCITY": {1: 2, 2: 0, 3: 1, 4: 0, 5: 0}
}
normal_relabel_rules = {
    "OEM": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7},
    "DFC19": {2: 0, 5: 1, 6: 2, 9: 3, 17: 4, 65: 2},
    "ISPRS": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "SYNTCITY": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
}


def labelmap(mask, rules):
    # Initialize mapped_mask with default value of 255, ensuring the same shape as mask
    mapped_mask = np.full(mask.shape, 255, dtype=np.uint8)
    
    # Iterate through each rule in the dictionary
    for src_value, dst_value in rules.items():
        # Ensure src_value is an integer, in case it's not already
        src_value = int(src_value)
        
        # Apply the mapping where mask values match the current rule's source value
        mapped_mask[mask == src_value] = dst_value
    
    return mapped_mask

def get_dataset_category(dataset_name):
    if dataset_name.issubset(ss_datasetname):
        if dataset_name.issubset(synrs3d_datasetname):
            return 'OEM'
        elif dataset_name.issubset(OEM_datasetname):
            return 'OEM'
        elif dataset_name.issubset(ISPRS_datasetname):
            return 'ISPRS'
        elif dataset_name.issubset(real_ss_datasetname) and not dataset_name.issubset(ISPRS_datasetname):
            return 'DFC19'
    return None

