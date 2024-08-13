import os.path as osp
import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from utils.templates import DATASETs_DOMAINs

@DATASET_REGISTRY.register()
class MS_DomainNet(DatasetBase):

    dataset_dir = "DomainNet"
    domains = DATASETs_DOMAINs[dataset_dir]["domains"]
    domain_ids = DATASETs_DOMAINs[dataset_dir]["domain_ids"]
    ids_domain = DATASETs_DOMAINs[dataset_dir]["ids_domain"]
    # domain_ids = {"clipart": 0, "painting": 1, "real": 2, "sketch": 3}
    # "ids_domain": {0: "clipart", 1: "painting", 2: "real", 3: "sketch"}

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.split_fewshot_path = os.path.join(self.dataset_dir, "splits")

        self.num_shots = cfg.DATASET.NUM_SHOTS
        self.source_domain = cfg.DATASET.SOURCE_DOMAINS  # list_name
        self.target_domain = cfg.DATASET.TARGET_DOMAINS

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x, train_u = self._read_data_from_txt(self.source_domain)

        test = self._read_data_from_txt(self.target_domain)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data_from_txt(self, input_domains):
        if input_domains == self.source_domain:
            x_items, u_items = [], []
            for _, dname in enumerate(input_domains):
                domain_id = self.domain_ids[dname]
                source_label_file = os.path.join(self.split_fewshot_path,
                                                 f"{dname}_labeled_{self.num_shots}.txt")
                if os.path.exists(source_label_file):
                    with open(source_label_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                impath=os.path.join(self.dataset_dir, image_path),
                                label=int(label),
                                domain=domain_id,
                                classname=str(class_name).lower()
                            )
                            x_items.append(item)

                source_unlabel_file = os.path.join(self.split_fewshot_path,
                                                   f"{dname}_unlabeled_{self.num_shots}.txt")
                if os.path.exists(source_unlabel_file):
                    with open(source_unlabel_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                impath=os.path.join(self.dataset_dir, image_path),
                                label=-1,  # int(label),
                                domain=domain_id,
                                classname=str(class_name).lower()
                            )
                            u_items.append(item)

            return x_items, u_items

        elif input_domains == self.target_domain:
            dname = input_domains[0]
            domain_id = self.domain_ids[dname]
            items = []
            target_file = os.path.join(self.split_fewshot_path, f"{dname}.txt")
            if os.path.exists(target_file):
                with open(target_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        text_list = line.split()
                        image_path, label = text_list[0], text_list[1]
                        class_name = image_path.split('/')[1]
                        item = Datum(
                            impath=os.path.join(self.dataset_dir, image_path),
                            label=int(label),
                            domain=domain_id,
                            classname=str(class_name).lower()
                        )
                        items.append(item)

            return items
