-naive DCGAN  
python main.py  

-conditional DCGAN
python main.py -c

-evaluation  
python main.py -e --netG uploaded_model/netG_epoch_74.pth

-load pretrain
python main.py -p --netG {netG_path} --netD {netD_path}
