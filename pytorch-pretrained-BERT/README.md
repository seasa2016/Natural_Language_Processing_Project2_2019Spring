rewrite from the bert model

to train:
python test_semeval.py --task_name semeval --do_train --data_dir {folder to data} --bert_model bert-base-uncased --do_lower_case --output_dir {output_dir}

to test:
python test_semeval.py --task_name semeval --do_test --data_dir {input file} --test_task taska --bert_model {model fir} --do_lower_case --output_dir {output_dir}
