# Domain Adaptative Retinal Image Quality Assessment with Knowledge Distillation Using Competitive Teacher-Student Network (ISBI 2023)

This repository contains the code for the paper Domain Adaptative Retinal Image Quality Assessment with Knowledge Distillation Using Competitive Teacher-Student Network (ISBI 2023).

![Structure.png](image%2FStructure.png)

## Datasets

The first dataset EyeQ is available at [EyeQ]([https://drive.grand-challenge.org/](https://github.com/hzfu/EyeQ)).
The second dataset DRIMDB is available at [DRIMDB](http://isbb.ktu.edu.tr/multimedia/drimdb).
All of the above datasets should be organized in the following structure:

```
Kaggle_DR_dataset  # data
    dataset_name
    -0  
       -image.png
...
 data              # label
    label.csv
...
```

where the `image.png` is the original fundus color image, `label.csv` is the ground truth of original fundus color image.

We select 169 images of EyeQ and DRIMDB datasets respectively and label them into two classes(Good and bad).

## Dependencies

* torch>=0.4.1
* torchvision>=0.2.1

## Acknowledgement

This work was supported in part by Basic and Applied Fundamental Research Foundation of Guangdong Province (2020A1515110286), The National Natural Science Foundation of China (8210072776), Guangdong Provincial Department of Education (2020ZDZX3043), Guangdong Provincial Key Laboratory (2020B121201001), and Shenzhen Natural Science Fund (JCYJ20200109140820699, 20200925174052004).

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{lin2023self,
  title={Domain Adaptative Retinal Image Quality Assessment with Knowledge Distillation Using Competitive Teacher-Student Network},
  author={Lin, Yuanming and Li, Heng and Liu, Haofeng and Shu, Hai and Li, Zinan and Hu, Yan and Liu, Jiang},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
