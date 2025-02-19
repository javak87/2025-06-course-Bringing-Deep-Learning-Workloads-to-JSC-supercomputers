---
author: Alexandre Strube // Sabrina Benassou // José Ignacio Robledo
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Data loading
date: March 19th, 2025
---

### Schedule for day 2

| Time          | Title                |
| ------------- | -----------          |
| 10:00 - 10:15 | Welcome, questions   |
| 10:15 - 11:30 | Data loading |
| 11:30 - 12:00 | Coffee Break (flexible) |
| 12:30 - 14:00 | Parallelize Training |

---

## Let's talk about DATA

![](images/data.jpeg)

--- 

## I/O is separate and shared

- All compute nodes of all supercomputers see the same files
- Performance tradeoff between shared acessibility and speed
- Our I/O server is almost a supercomputer by itself
    ![JSC Supercomputer Stragegy](images/machines.png){height=350pt}

---

## Where do I keep my files?

- Always store your code in the project folder (**`$PROJECT_projectname`** ). In our case 

    ```bash
    /p/project/training2449/$USER
    ```

- Store data in the scratch directory for faster I/O access (**`$SCRATCH_projectname`**). Files in scratch are deleted after 90 days of inactivity.
    
    ```bash
    /p/scratch/training2449/$USER
    ```

- Store the data in [`$DATA_dataset`](https://judoor.fz-juelich.de/projects/datasets/) for a more permanent location. **This location is not accessible by compute nodes. Please copy the data to scratch to ensure your job can access it**.

    ```bash
    /p/data1/datasets
    ```

---

## Data loading

- We have CPUs and lots of memory - let's use them
- If your dataset is relatively small (< 500 GB) and can fit into the working memory (RAM) of each compute node (along with the program state), you can store it in **``/dev/shm``**. This is a special filesystem that uses RAM for storage, making it extremely fast for data access. ⚡️
- For bigger datasets (> 500 GB) you have many strategies:
    - Hierarchical Data Format 5 (HDF5)
    - Apache Arrow
    - NVIDIA Data Loading Library (DALI)
    - SquashFS

---

## Data loading

- In this course, we will demonstrate how to store your data in HDF5 and PyArrow files.
- We will use the ImageNet dataset to create HDF5 and PyArrow files.

---

## But before

- We need to download some code

    ```bash
    cd $HOME/course
    git clone https://github.com/HelmholtzAI-FZJ/2025-03-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers.git
    ```

- Move to the correct folder

    ```
    cd 2025-03-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers/code/dataloading/
    ```

---

## The ImageNet dataset
#### Large Scale Visual Recognition Challenge (ILSVRC)
- An image dataset organized according to the [WordNet hierarchy](https://wordnet.princeton.edu). 
- Extensively used in algorithms for object detection and image classification at large scale. 
- It has 1000 classes, that comprises 1.2 million images for training, and 50,000 images for the validation set.

![](images/imagenet_banner.jpeg)

---

## The ImageNet dataset

```bash
ILSVRC
|-- Data/
    `-- CLS-LOC
        |-- test
        |-- train
        |   |-- n01440764
        |   |   |-- n01440764_10026.JPEG
        |   |   |-- n01440764_10027.JPEG
        |   |   |-- n01440764_10029.JPEG
        |   |-- n01695060
        |   |   |-- n01695060_10009.JPEG
        |   |   |-- n01695060_10022.JPEG
        |   |   |-- n01695060_10028.JPEG
        |   |   |-- ...
        |   |...
        |-- val
            |-- ILSVRC2012_val_00000001.JPEG  
            |-- ILSVRC2012_val_00016668.JPEG  
            |-- ILSVRC2012_val_00033335.JPEG      
            |-- ...
```
---

## The ImageNet dataset
imagenet_train.pkl

```bash 
{
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_8050.JPEG': 524,
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_12728.JPEG': 524,
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_9736.JPEG': 524,
    ...
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_7460.JPEG': 524,
    ...
 }
```

imagenet_val.pkl

```bash
{
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00008838.JPEG': 785,
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00008555.JPEG': 129,
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00028410.JPEG': 968,
    ...
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00016007.JPEG': 709,
 }
```

---

## Access File System

```python
class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):

        file_name = "imagenet_train.pkl" if split == "train" else "imagenet_val.pkl"

        with open(os.path.join(args.data_root, file_name), "rb") as f:
            train_data = pickle.load(f)

        self.samples = list(train_data.keys())
        self.targets = list(train_data.values())
        self.transform = transform

    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.samples[idx])).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]    
   
```

---

## Inodes 
- Inodes (Index Nodes) are data structures that store metadata about files and directories.
- Unique identification of files and directories within the file system.
- Efficient management and retrieval of file metadata.
- Essential for file operations like opening, reading, and writing.
- **Limitations**:
  - **Fixed Number**: Limited number of inodes; no new files if exhausted, even with free disk space.
  - **Space Consumption**: Inodes consume disk space, balancing is needed for efficiency.
![](images/inodes.png)

---

## Pyarrow File Creation

![](images/field.png)

```python 
    binary_t = pa.binary()
    uint16_t = pa.uint16()
```

---

## Pyarrow File Creation

![](images/schema.png)

```python 
    binary_t = pa.binary()
    uint16_t = pa.uint16()

    schema = pa.schema([
        pa.field('image_data', binary_t),
        pa.field('label', uint16_t),
    ])
```

---

## Pyarrow File Creation

![](images/file.png){width=700 height=350}

```python 
    with pa.OSFile(
            os.path.join(args.target_folder, f'ImageNet-{split}.arrow'),
            'wb',
    ) as f:
        with pa.ipc.new_file(f, schema) as writer:
```

---

## Pyarrow File Creation

![](images/batch.png){width=650 height=300}

```python 
    for (sample, label) in tqdm(zip(samples, targets)):
        with open(os.path.join(args.data_root, sample), 'rb') as f:
            img_string = f.read()

        image_data = pa.array([img_string], type=binary_t)
        label = pa.array([label], type=uint16_t)

        batch = pa.record_batch([image_data, label], schema=schema)

        writer.write(batch)
```

---

## Pyarrow File Creation

![](images/pyarrow.png){width=650 height=300}

```python 

    for (sample, label) in tqdm(zip(samples, targets)):
        with open(os.path.join(args.data_root, sample), 'rb') as f:
            img_string = f.read()

        image_data = pa.array([img_string], type=binary_t)
        label = pa.array([label], type=uint16_t)

        batch = pa.record_batch([image_data, label], schema=schema)

        writer.write(batch)
```

---

## Access Arrow File

::: {.container}
:::: {.col}
![](images/pyarrow.png){width=500 height=300}
::::
:::: {.col}
```python
def __getitem__(self, idx):
    if self.arrowfile is None:
        self.arrowfile = pa.OSFile(self.data_root, 'rb')
        self.reader = pa.ipc.open_file(self.arrowfile)

    row = self.reader.get_batch(idx)

    img_string = row['image_data'][0].as_py()
    target = row['label'][0].as_py()

    with io.BytesIO(img_string) as byte_stream:
        with Image.open(byte_stream) as img:
            img = img.convert("RGB")

    if self.transform:
        img = self.transform(img)

    return img, target

```
::::
:::

---

## HDF5

![](images/h5.png)

```python

with h5py.File(os.path.join(args.target_folder, 'ImageNet.h5'), "w") as f:

```

---

## HDF5

::: {.container}
:::: {.col}
```python

group = g.create_group(split)

```
::::
:::: {.col}
![](images/groups.png)
::::
:::

---

## HDF5


::: {.container}
:::: {.col}
``` python 
dt_sample = h5py.vlen_dtype(np.dtype(np.uint8))
dt_target = np.dtype('int16')

dset = group.create_dataset(
                'images',
                (len(samples),),
                dtype=dt_sample,
            )

dtargets = group.create_dataset(
        'targets',
        (len(samples),),
        dtype=dt_target,
    )
```
::::
:::: {.col}
![](images/datasets.png){width=400 height=350}
::::
:::

---

## HDF5


![](images/first_iter.png){width=750 height=350}

```python
for idx, (sample, target) in tqdm(enumerate(zip(samples, targets))):        
    with open(sample, 'rb') as f:
        img_string = f.read() 
        dset[idx] = np.array(list(img_string), dtype=np.uint8)
        dtargets[idx] = target
```

---

## HDF5


![](images/last_iter.png){width=750 height=350}

```python
for idx, (sample, target) in tqdm(enumerate(zip(samples, targets))):        
    with open(sample, 'rb') as f:
        img_string = f.read() 
        dset[idx] = np.array(list(img_string), dtype=np.uint8)
        dtargets[idx] = target
```

---

## HDF5


![](images/hdf5.png)

---

## Access h5 File 

```python
def __getitem__(self, idx):
    if self.h5file is None:
        self.h5file = h5py.File(self.train_data_path, 'r')[self.split]
        self.imgs = self.h5file["images"]
        self.targets = self.h5file["targets"]

    img_string = self.imgs[idx]
    target = self.targets[idx]

    with io.BytesIO(img_string) as byte_stream:
        with Image.open(byte_stream) as img:
            img = img.convert("RGB")

    if self.transform:
        img = self.transform(img)
        
    return img, target
```

---

## DEMO

---

## Exercise

- Could you create an arrow file for the flickr dataset stored in 
```/p/scratch/training2449/data/Flickr30K/```
and read it using a dataloader ?
