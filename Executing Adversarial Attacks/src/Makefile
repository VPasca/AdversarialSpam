TRAIN=../data/train.tsv
TEST=../data/test.tsv
TESTLSTM=../data/TestForLSTM/test.tsv
	
train_lstm:
	python train_LSTM.py --train_path $(TRAIN) --test_path $(TEST) #--model_path ../model/model_lstm

train_cnn:
	python train_CNN.py --train_path $(TRAIN) --test_path $(TEST) #--model_path ../model/model_cnn

attack_lstm:
	python3 discrete_attack.py 2 LSTM $(TRAIN) $(TESTLSTM) ../data/changed_lstm --embedding_path ../paragram_300_sl999/Trying\ Again/paragram_300_sl999\(1\)/paragram_300_sl999/altered\ \(add\ header\)/paragram_300_sl999.txt --model_path ../model/model_lstm 20000 --first_label REAL #FAKE

attack_cnn:
	python3 discrete_attack.py 2 CNN $(TRAIN) $(TEST) ../data/changed_cnn_train --embedding_path ../paragram_300_sl999/Trying\ Again/paragram_300_sl999\(1\)/paragram_300_sl999/altered\ \(add\ header\)/paragram_300_sl999.txt   --model_path ../model/model_cnn 20000 --first_label REAL #FAKE

attack_lstm_joint:
	python3 joint_attack.py 2 2 LSTM $(TRAIN) $(TEST) ../data/changed_lstm --embedding_path ../paragram_300_sl999/Trying\ Again/paragram_300_sl999\(1\)/paragram_300_sl999/altered\ \(add\ header\)/paragram_300_sl999.txt --model_path ../model/model_lstm.pt 20000 --first_label FAKE

attack_cnn_joint:
	python3 joint_attack.py 2 2 CNN $(TRAIN) $(TEST) ../data/changed_cnn_train --embedding_path ../paragram_300_sl999/Trying\ Again/paragram_300_sl999\(1\)/paragram_300_sl999/altered\ \(add\ header\)/paragram_300_sl999.txt  --model_path ../model/model_cnn.pt 20000 --first_label FAKE

eval_cnn:
	python3 evaluate.py --train_path $(TRAIN) --test_path $(TEST) --model_path ../model/model_cnn


