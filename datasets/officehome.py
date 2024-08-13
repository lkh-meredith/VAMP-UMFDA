import os.path as osp
import os
from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from utils.templates import DATASETs_DOMAINs


@DATASET_REGISTRY.register()
class MS_OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "OfficeHome"
    domains = DATASETs_DOMAINs[dataset_dir]["domains"]#["art", "clipart", "product", "real_world"]
    domain_ids = DATASETs_DOMAINs[dataset_dir]["domain_ids"]
    ids_domain = DATASETs_DOMAINs[dataset_dir]["ids_domain"]

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
                if dname == "Real World":
                    file_dname = "Real"
                else:
                    file_dname = dname
                domain_id = self.domain_ids[dname]
                source_label_file = os.path.join(self.split_fewshot_path, f"{file_dname}_labeled_p0{self.num_shots}.txt")
                if os.path.exists(source_label_file):
                    with open(source_label_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                        impath=os.path.join(self.dataset_dir,image_path),
                                        label=int(label),
                                        domain=domain_id,
                                        classname=str(class_name).lower()
                                    )
                            x_items.append(item)

                source_unlabel_file = os.path.join(self.split_fewshot_path, f"{file_dname}_unlabeled_p0{self.num_shots}.txt")
                if os.path.exists(source_unlabel_file):
                    with open(source_unlabel_file,"r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                impath=os.path.join(self.dataset_dir, image_path),
                                label=-1, #int(label),
                                domain=domain_id,
                                classname=str(class_name).lower()
                            )
                            u_items.append(item)

            return x_items, u_items

        elif input_domains == self.target_domain:
            dname = input_domains[0]
            if dname == "Real World":
                file_dname = "Real"
            else:
                file_dname = dname
            domain_id = self.domain_ids[dname]
            items = []
            target_file = os.path.join(self.split_fewshot_path, f"{file_dname}.txt")
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


    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)

        return items
