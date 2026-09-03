#python scripts/run_experiment.py --peft lora --fl fedavg --alpha 0.1
#python scripts/run_experiment.py --peft lora --fl fedavg --alpha 1
#python scripts/run_experiment.py --peft lora --fl fedavg --alpha 10
python scripts/run_experiment.py --peft lora --fl fedprox --alpha 0.1
python scripts/run_experiment.py --peft lora --fl fedprox --alpha 1
python scripts/run_experiment.py --peft lora --fl fedprox --alpha 10

python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedavg --alpha 0.1
python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedavg --alpha 1
python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedavg --alpha 10
python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedprox --alpha 0.1
python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedprox --alpha 1
python scripts/run_experiment.py --peft qlora --qlora-bits 8 --fl fedprox --alpha 10

python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 1
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedavg --alpha 10

python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedprox --alpha 0.1
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedprox --alpha 1
python scripts/run_experiment.py --peft qlora --qlora-bits 4 --fl fedprox --alpha 10
