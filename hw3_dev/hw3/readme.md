-naive DCGAN  
python hw3_1_main.py  

-conditional DCGAN  
python hw3_1_main.py -c  

-WGAN  
python hw3_1_main.py -w  

-conditional WGAN  
python hw3_1_main.py -c -w  

-evaluation  
python hw3_1_main.py -e --netG stored_model/netG_epoch_21_100  

-load pretrain  
python hw3_1_main.py -p --netG {netG_path} --netD {netD_path}  
