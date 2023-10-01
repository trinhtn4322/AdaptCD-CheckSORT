import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class MyDataset(BaseDataset):
    CLASSES = ["00001","00002","00003","00004","00005","00006","00007",
               "00008","00009","00010","00011","00012","00013","00014",
               "00015","00016","00017","00018","00019","00020","00021",
               "00022","00023","00024","00025","00026","00027","00028",
               "00029","00030","00031","00032","00033","00034","00035",
               "00036","00037","00038","00039","00040","00041","00042",
               "00043","00044","00045","00046","00047","00048","00049",
               "00050","00051","00052","00053","00054","00055","00056",
               "00057","00058","00059","00060","00061","00062","00063",
               "00064","00065","00066","00067","00068","00069","00070",
               "00071","00072","00073","00074","00075","00076","00077",
               "00078","00079","00080","00081","00082","00083","00084",
               "00085","00086","00087","00088","00089","00090","00091",
               "00092","00093","00094","00095","00096","00097","00098",
               "00099","00100","00101","00102","00103","00104","00105",
               "00106","00107","00108","00109","00110","00111","00112",
               "00113","00114","00115","00116","00117"]
    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
