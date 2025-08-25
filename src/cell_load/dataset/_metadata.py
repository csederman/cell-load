from torch.utils.data import ConcatDataset, Dataset


class MetadataConcatDataset(ConcatDataset):
    """
    ConcatDataset that enforces consistent metadata across all constituent datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)

        # Get the base dataset, handling both Subset and MixupPerturbationDataset
        first_ds = datasets[0]
        if hasattr(first_ds, "dataset"):
            # This is a Subset
            self.base = first_ds.dataset  # type: ignore
        elif hasattr(first_ds, "_pd"):
            # This is a MixupPerturbationDataset
            self.base = first_ds._pd  # type: ignore
        else:
            # This is a direct PerturbationDataset
            self.base = first_ds

        self.embed_key = self.base.embed_key  # type: ignore
        self.control_pert = self.base.control_pert  # type: ignore
        self.pert_col = self.base.pert_col  # type: ignore
        self.batch_col = self.base.batch_col  # type: ignore

        for ds in datasets:
            # Get the underlying PerturbationDataset for each dataset
            if hasattr(ds, "dataset"):
                # This is a Subset
                md = ds.dataset  # type: ignore
            elif hasattr(ds, "_pd"):
                # This is a MixupPerturbationDataset
                md = ds._pd  # type: ignore
            else:
                # This is a direct PerturbationDataset
                md = ds

            if (
                md.embed_key != self.embed_key  # type: ignore
                or md.control_pert != self.control_pert  # type: ignore
                or md.pert_col != self.pert_col  # type: ignore
                or md.batch_col != self.batch_col  # type: ignore
            ):
                raise ValueError(
                    "All datasets must share the same embed_key, control_pert, pert_col, and batch_col"
                )
