# DCGAN
### naive DCGAN  
* train  
`python main.py ` 
* inference  
`python main.py -e --netG stored_model/netG_epoch_21_100`  
* load pretrain  
`python main.py -p --netG {netG} --netD {netD}`  

### conditional DCGAN  
* train  
`python main.py -c`
* inference  
`python main.py -c -e --ec {tag.txt} --netG stored_model/netG_epoch_17con`
* load pretrain  
`python main.py -c -p --netG {netG} --netD {netD}`  

# WGAN  
### naive WGAN  
* train  
`python main.py -w`  
* inference  
`python main.py -w -e --netG {model}`
* load pretrain  
`python main.py -w -p --netG {netG} --netD {netD}`  

### conditional WGAN  
* train  
`python main.py -c -w` 
* inference  
`python main.py -c -w -e --ec {tag.txt} --netG {model}`
* load pretrain  
`python main.py -c -w -p --netG {netG} --netD {netD}`  

# ACGAN  
### conditional  
* train  
`python acgan.py`  
