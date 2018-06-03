-naive DCGAN  
python main.py  
python main.py -e --netG stored_model/netG_epoch_21_100   

-conditional DCGAN  
python main.py -c
python main.py -c -e -ec {tag.txt} --netG stored_model/netG_epoch_17con

-WGAN  
python main.py -w  

-conditional WGAN  
python main.py -c -w  

-load pretrain  
python main.py -p --netG {netG_path} --netD {netD_path}  
