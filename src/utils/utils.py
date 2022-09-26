import numpy as np
AVAILABLE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False


class ResultFiles(object):
    def __init__(self, args):
        self.out_path = args.out_path
        self.start_trial = args.start_trial
        self.wandb = (not args.no_wandb) and AVAILABLE_WANDB

        # experimental settings file
        result_file = open(args.out_path + f'Setting_{args.data}_{args.al_model}_{args.start}.txt', 'w')
        result_file.write('='*30+'\n')
        for arg in vars(args):
            result_file.write(f' {arg} = {getattr(args, arg)}\n')
        result_file.write('='*30+'\n')
        result_file.close()

        # experimental results file
        self.result_filename = args.out_path + f'TestAcc_{args.data}_{args.al_model}_{args.start}.txt'
        self.result_file = open(self.result_filename, 'w')
        self.result_file.write('Trial,Cycle,nLabeled,TestAcc\n')
        self.result_file.close()

        # result table for wandb update
        self.test_accs = []

        # wandb result
        if self.wandb:
            wandb.init(
                project=f'{args.data}',
                name=f"{args.al_model}",
                config=args,
                dir='.',
                save_code=True
            )
            wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or x == 'main.py')

    
    def new_trial_result(self, trial):
        self.trial = trial
        self.test_acc_trial = []
        # experimental results file
        self.result_file = open(self.result_filename, 'a')
        # experimental results of queried index file
        index_file_name = self.out_path + f'Idx/TIdx{str(self.start_trial + trial + 1).zfill(2)}.txt'
        self.result_idx_file = open(index_file_name, 'w')
    
    def end_trial_result(self):
        self.result_idx_file.close()
        self.result_file.close()
        self.test_accs.append(self.test_acc_trial)
        self.test_acc_trial = []
    
    def save_query_index(self, indices):
        indices.tofile(self.result_idx_file, sep='\n')
        self.result_idx_file.write('\n')
    
    def save_result(self, idxs_lb, test_acc, round):
        # experimental results file
        self.result_file.write('{},{},{},{:.4f}\n'.format(self.trial + 1, round + 1, sum(idxs_lb), test_acc['acc']))
        self.test_acc_trial.append(test_acc['acc'])
    
    def wandb_final_result(self):
        if self.wandb:
            test_accs = np.array(self.test_accs).mean(axis=0)
            for r, acc in enumerate(test_accs):
                wandb.log({'TestAcc': acc}, step=r)