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