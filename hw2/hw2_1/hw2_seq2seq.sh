wget -P stored_model https://www.dropbox.com/s/ejxj9qf9r0zwwq4/decoder
wget -P stored_model https://www.dropbox.com/s/o6nh33ucbo90ly4/encoder
python3 model_seq2seq.py $1 $2