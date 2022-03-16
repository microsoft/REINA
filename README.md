# REINA
Implementation of the following paper:
## Training Data is More Valuable than You Think: A Simple and Effective Method by Retrieving from Training Data
Shuohang Wang (shuowa@microsoft.com), Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu, Chenguang Zhu, Michael Zeng



Accept to ACL2022 main conference

### Usage 1
After cloning the repo, run the following code with docker to reproduce REINA on XSum dataset. REINA is interaged into the model trainig code.  Please set model name to google/pegasus-large or facebook/bart-large or facebook/bart-base, etc. By default, the job is run on 8 GPUs. Please tuning "--gradient_accumulation_steps" if use less GPUs. More --reina_workers is prefered to speed up REINA process. 40 workers will task around 15 minutes. 
```
docker run --gpus all -it --rm --shm-size 10g -w /home/reina/src -v ${PWD}/REINA:/home/reina shuohang/pytorch:reina /bin/bash -c "export HF_DATASETS_CACHE=/home/reina/data; export TRANSFORMERS_CACHE=/home/reina/cache; python -m torch.distributed.launch --nproc_per_node=8 run_summarization.py --report_to none  --save_strategy epoch --model_name_or_path google/pegasus-large --dataset_name xsum  --do_train   --do_eval --do_predict  --per_device_train_batch_size=2 --gradient_accumulation_steps 2 --per_device_eval_batch_size=4 --predict_with_generate --output_dir /home/reina/output --overwrite_output_dir --text_column document --summary_column summary  --num_train_epochs 3 --logging_strategy epoch --evaluation_strategy epoch --load_best_model_at_end --max_target_length 64 --val_max_target_length 64 --learning_rate 0.00005 --reina --reina_workers 40"
```

### Usage 2
In this section, the REINA and model training are splitted in two steps. The first step will save REINA data into files and then run seq2seq model for summarization.
```
docker run --gpus all -it --rm --shm-size 10g -w /home/reina/src -v ${PWD}/REINA:/home/reina shuohang/pytorch:reina /bin/bash -c "export HF_DATASETS_CACHE=/home/reina/data; python reina.py --dataname xsum --reina_workers 10 --key_column document --value_column summary"
docker run --gpus all -it --rm --shm-size 10g -w /home/reina/src -v ${PWD}/REINA:/home/reina shuohang/pytorch:reina /bin/bash -c "export HF_DATASETS_CACHE=/home/reina/data; export TRANSFORMERS_CACHE=/home/reina/cache; python -m torch.distributed.launch --nproc_per_node=8 run_summarization.py --report_to none  --save_strategy epoch --model_name_or_path google/pegasus-large --dataset_name xsum  --do_train   --do_eval --do_predict  --per_device_train_batch_size=2 --gradient_accumulation_steps 2 --per_device_eval_batch_size=4 --predict_with_generate --output_dir /home/reina/output --overwrite_output_dir --text_column document --summary_column summary  --num_train_epochs 3 --logging_strategy epoch --evaluation_strategy epoch --load_best_model_at_end --max_target_length 64 --val_max_target_length 64 --learning_rate 0.00005  --train_file /home/reina/data/reina/xsum/train.json --validation_file /home/reina/data/reina/xsum/validation.json --test_file /home/reina/data/reina/xsum/test.json"
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
