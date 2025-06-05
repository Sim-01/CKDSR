"main_KD" is the main program, which mainly includes three parts: training model, testing synthetic data, and testing field data.
''option" is a parameter configuration module that uses Python's argparse library to parse command line parameters.
"utility" is a training tool module, which includes the following main functions: record time, save models, pictures, test results, etc.
"trainer" defines two implementations of trainer classes: knowledge distillation (Trainer) and single model training (Trainer_single).
