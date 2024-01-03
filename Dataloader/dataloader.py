import os
import numpy as np
import torch
import glob
import nibabel as niba
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class OrganSegmentationDataset(Dataset):

    def __init__(
            self,
            src='/content/drive/MyDrive/UNETR/Dummy_dataset/',
            # images_dir="/content/drive/MyDrive/Research/Image_segmentation/processed_data",
            # images_dir="/content/drive/MyDrive/dataset/processed_data",
            subset="train"
    ):
        self.subset = subset
        if self.subset == "train":
            self.images_dir = "/content/drive/MyDrive/UNETR/SegThor_3D/train"
        else:
            self.images_dir = "/content/drive/MyDrive/UNETR/SegThor_3D/test"

        self.image_Paths = []
        self.patient_ids = []
        self.required_test = False
        assert subset in ["all", "train", "validation"]

        print("reading {} images...".format(subset))
        if (subset == "train"):
            self.image_Paths = sorted(glob.glob(self.images_dir + "/image/*.nii.gz"))
            for image_Paths in self.image_Paths:
                patient_id = int(image_Paths.split(os.sep)[-1].split("_")[1].split(".")[0])
                if patient_id not in self.patient_ids:
                    self.patient_ids.append(patient_id)

        elif (subset == "validation"):
            self.image_Paths = sorted(glob.glob(self.images_dir + "/image/*.nii.gz"))
            for image_Path in self.image_Paths:
                patient_id = int(image_Path.split(os.sep)[-1].split("_")[1].split(".")[0])
                if patient_id not in self.patient_ids:
                    self.patient_ids.append(patient_id)

        print(self.patient_ids)

    def __len__(self):
        return len(self.image_Paths)

    def __getitem__(self, id):
        max_axial_size = 128
        filePath = self.image_Paths[id]
        patient_id = filePath.split(os.sep)[-1].split("_")[1].split(".")[0]
        image_nifti = niba.load(filePath)
        image = image_nifti.get_fdata()

        mask_nifti = niba.load(self.images_dir + f"/gt/P_{patient_id}_GT.nii.gz")
        mask = mask_nifti.get_fdata()

        image_min = np.amin(image)

        if image_min < 0:
            image = image + image_min

        image_max = np.amax(image)
        image = image / image_max

        _, _, axial_size = image.shape
        if axial_size < max_axial_size:
            extra = max_axial_size - axial_size
            extra_slice = np.zeros((256, 256, extra))
            image = np.concatenate((image, extra_slice), axis=2)
            mask = np.concatenate((mask, extra_slice), axis=2)
        elif axial_size > max_axial_size:
            image = image[:, :, :128]
            mask = mask[:, :, :128]

        # image = image[:128,:128,:]
        # mask = mask[:128,:128,:]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(int))

        return image_tensor, mask_tensor, patient_id


def data_loaders():
    dataset_train, dataset_valid = datasets()

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets():
    train = OrganSegmentationDataset(
        subset="train"
    )
    valid = OrganSegmentationDataset(
        subset="validation"
    )
    return train, valid
