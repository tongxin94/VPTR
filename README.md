**VPTR**: End-to-End Real-Time Vanishing Point Detection with Transformer
========

1. Install the requirements from requirements.txt
```
pip install -r requirements.txt
```

2. Download the pretrained model from:
[SU3_Pretrained_Model](https://drive.google.com/file/d/1hKmYDe10RVDnVvLVQPc4ESsd95duF-lV/view).
 
3. Download the SU3 dataset

4. Testing the model on SU3
```
python train_vp.py  --batch_size 4 --num_workers=4 --no_aux_loss --test --resume path/to/pretrained_model
```
5. Part of the code comes from :
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
6. If you find the work helpful for your research, please consider citing it
```
@inproceedings{tong2024end,
  title={End-to-end real-time vanishing point detection with transformer},
  author={Tong, Xin and Peng, Shi and Guo, Yufei and Huang, Xuhui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5243--5251},
  year={2024}
}
```
