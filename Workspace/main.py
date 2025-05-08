import os
from torch.utils.data import DataLoader
from dataset_definition import ImageGraphDataset

def collate_graphs(batch):
    return batch

def get_dataloader(
    image_array,
    batch_size: int = 1,
    shuffle: bool = True,
    segmenter: str = 'slic',
    **seg_kwargs
):
    dataset = ImageGraphDataset(
        images=image_array,
        segmenter=segmenter,
        **seg_kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs
    )

if __name__ == '__main__':
    from skimage import io

    img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
    # imgs = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)][:10]
    imgs = [io.imread(os.path.join(img_dir, "ISIC_0015719.jpg"))]

    loader = get_dataloader(
        image_array=imgs,
        batch_size=2,
        segmenter='slic',
        n_segments=100
    )
    for batch in loader:
        for G in batch:
            print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
