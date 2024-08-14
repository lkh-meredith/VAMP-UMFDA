import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from dassl.data.datasets import build_dataset
# from .samplers import build_sampler
from dassl.data.transforms.transforms import INTERPOLATION_MODES, build_transform
from dassl.data.samplers import RandomSampler, SequentialSampler, RandomDomainSampler, SeqDomainSampler, RandomClassSampler
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random

class MyDomainSampler(Sampler):
    """Randomly samples one domains each with N annotated images and M unannotated image
    """

    def __init__(self, data_source, batch_size_x, batch_size_u):
        self.labeled_data = data_source[0]
        self.unlabeled_data = data_source[1]

        idx = 0
        # self.all_data = []
        self.domain_indices = defaultdict(lambda: defaultdict(list))
        for i, item in enumerate(self.labeled_data):
            # print(item.domain)
            self.domain_indices['labeled'][item.domain].append(i)
            # self.all_data.append(item)
            idx += 1

        start_idx_u = idx
        for i, item in enumerate(self.unlabeled_data):
            self.domain_indices['unlabeled'][item.domain].append(i + start_idx_u)
            # self.all_data.append(item)
            idx += 1

        assert idx == (len(self.labeled_data) + len(self.unlabeled_data))
      
        self.batch_size_x = batch_size_x
        self.batch_size_u = batch_size_u

        self.domains = list(self.domain_indices['labeled'].keys())
        domain_labeled_num = [len(self.domain_indices['labeled'][item]) for item in self.domains]
        for i, j in zip(self.domains, domain_labeled_num):
            print(f"The number of labeld data on source {i}: {j}")

        min_domain_labeled_num = min(domain_labeled_num)
        print(f"The minimum number of the labeled data: {min_domain_labeled_num}")

        self.length = min_domain_labeled_num // self.batch_size_x  # the number of iteriations in a epoch is calculated using annotated source data.

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_indices)
    
        while True:
            selected_domains = random.sample(self.domains, 1)
            selected_domains = selected_domains[0]
            label_idxs = domain_dict['labeled'][selected_domains] 
            unlabel_idxs = domain_dict['unlabeled'][selected_domains]

            batch_x_idx = random.sample(label_idxs, self.batch_size_x)
            batch_u_idx = random.sample(unlabel_idxs, self.batch_size_u)

            batch = batch_x_idx + batch_u_idx

            yield batch

            for idx in batch_x_idx:
                domain_dict['labeled'][selected_domains].remove(idx)

            for idx in batch_u_idx:
                domain_dict['unlabeled'][selected_domains].remove(idx)

            remaining_x = len(domain_dict['labeled'][selected_domains])
            remaining_u = len(domain_dict['unlabeled'][selected_domains]) 

            if remaining_x < self.batch_size_x:
                break
            if remaining_u < self.batch_size_u:
                domain_dict['unlabeled'] = copy.deepcopy(self.domain_indices['unlabeled'])


    def __len__(self):
        return self.length


def build_sampler(
    sampler_type,
    cfg=None,
    data_source=[],
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if len(data_source)<=1:
        data_source = data_source[0]

    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    elif sampler_type == "RandomDomainSampler":
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == "SeqDomainSampler":
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == "RandomClassSampler":
        return RandomClassSampler(data_source, batch_size, n_ins)

    elif sampler_type == "MyDomainSampler":
        return MyDomainSampler(data_source, cfg.DATALOADER.TRAIN_X.BATCH_SIZE, cfg.DATALOADER.TRAIN_U.BATCH_SIZE)
    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))



def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    if not is_train:
        data_source = data_source[0]

    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:

        dataset_wrapper = DatasetWrapper

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )

    elif dataset_wrapper == "dataset_wrapper_xu":
        dataset_wrapper = DatasetWrapperXU

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_sampler = sampler,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )

    assert len(data_loader) > 0

    return data_loader


class MyDataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=[dataset.train_x, dataset.train_u],
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper="dataset_wrapper_xu"
        )

       
        train_loader_u = None

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=[dataset.val],
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=[dataset.test],
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

class DatasetWrapperXU(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        # self.data_source = data_source
        self.labeled_data = data_source[0]
        self.unlabeled_data = data_source[1]

        idx = 0
        self.all_data = []
        self.domain_indices = defaultdict(lambda: defaultdict(list))
        for i, item in enumerate(self.labeled_data):
            self.domain_indices['labeled'][item.domain].append(i)
            self.all_data.append(item)
            idx += 1

        start_idx_u = idx
        for i, item in enumerate(self.unlabeled_data):
            self.domain_indices['unlabeled'][item.domain].append(i+start_idx_u)
            self.all_data.append(item)
            idx += 1

        assert idx == len(self.all_data)

        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.all_data) 

    def __getitem__(self, idx):
        item = self.all_data[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
