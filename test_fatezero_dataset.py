

from test_fatezero import *
from glob import glob
import copy

@click.command()
@click.option("--edit_config", type=str,    default="config/supp/style/0313_style_edit_warp_640.yaml")
@click.option("--dataset_config", type=str, default="data/supp_edit_dataset/dataset_prompt.yaml")
def run(edit_config, dataset_config):
    Omegadict_edit_config = OmegaConf.load(edit_config)
    Omegadict_dataset_config = OmegaConf.load(dataset_config)

    # Go trough all data sample
    data_sample_list = sorted(Omegadict_dataset_config.keys())
    print(f'Datasample to evaluate: {data_sample_list}')
    dataset_time_string = get_time_string()
    for data_sample in data_sample_list:
        print(f'Evaluate {data_sample}')
        # breakpoint()

        for p2p_config_index, p2p_config in Omegadict_edit_config['validation_sample_logger_config']['p2p_config'].items():
            edit_config_now = copy.deepcopy(Omegadict_edit_config)
            edit_config_now['train_dataset'] = copy.deepcopy(Omegadict_dataset_config[data_sample])
            edit_config_now['train_dataset'].pop('target')
            if 'eq_params' in edit_config_now['train_dataset']:
                edit_config_now['train_dataset'].pop('eq_params')
            # edit_config_now['train_dataset']['prompt'] = Omegadict_dataset_config[data_sample]['source']
            # breakpoint()
            
            edit_config_now['validation_sample_logger_config']['prompts'] \
                = copy.deepcopy( [Omegadict_dataset_config[data_sample]['prompt'],]+ OmegaConf.to_object(Omegadict_dataset_config[data_sample]['target']))
            p2p_config_now = dict()
            for i in range(len(edit_config_now['validation_sample_logger_config']['prompts'])):
                p2p_config_now[i] = p2p_config
                if 'eq_params' in Omegadict_dataset_config[data_sample]:
                    p2p_config_now[i]['eq_params'] = Omegadict_dataset_config[data_sample]['eq_params']
            
            edit_config_now['validation_sample_logger_config']['p2p_config'] = copy.deepcopy(p2p_config_now)
            edit_config_now['validation_sample_logger_config']['source_prompt'] = Omegadict_dataset_config[data_sample]['prompt']
            # edit_config_now['validation_sample_logger_config']['source_prompt'] = Omegadict_dataset_config[data_sample]['eq_params']
            
            
            # if 'logdir' not in edit_config_now:
            # breakpoint()
            logdir = edit_config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')+f'_config_{p2p_config_index}'+f'_{os.path.basename(dataset_config)[:-5]}'+f'_{dataset_time_string}'
            logdir +=  f"/{data_sample}"            
            edit_config_now['logdir'] = logdir
            print(f'Saving at {logdir}')
            
            # breakpoint()
            test(config=edit_config, **edit_config_now)


if __name__ == "__main__":
    run()
