import re

from pssh.clients import SSHClient
import tempfile
import time
import os

from . import Config, ClusterJobBaseArguments
from .files import FileHandler


class PythonEnvironment:
    def __init__(self, config: Config, ssh_client: SSHClient):
        self.config = config
        self.ssh_client = ssh_client

    def get_header(self):
        ret = '\n'
        # load the anaconda module
        ret += f'module load {self.config.anaconda}\n'
        # activate the environment
        ret += f'source activate {self.config.envdir_path}\n'
        return ret

    def get_run_command(self, script: str, kwargs: dict = None):
        args = f'--args-file={self.config.args_file}'
        if kwargs is not None:
            args = ' '.join([f' --{kw}={val}' for kw, val in kwargs.items()])
        return f'python {self.config.codedir_path}/{script} {args}\n'

    def prepare_environment(self):
        print('Prepping python environment...')
        with self.ssh_client.open_shell() as shell:
            # ensure we are using bash
            shell.run('/bin/bash')

            print('> load anaconda module')
            shell.run(f'module load {self.config.anaconda}')
            print('> create anaconda environment')
            shell.run(f'conda create --prefix={self.config.envdir_path} -y python={self.config.python}')
            print('> activate anaconda environment')
            shell.run(f'source activate {self.config.envdir_path}')
            print('> pip install requirements.txt')
            shell.run(f'pip install --no-input -r {os.path.join(self.config.workdir_path, "requirements.txt")}')
        print('\n'.join(list(shell.output.stdout)))
        print('\n'.join(list(shell.output.stderr)))


class ClusterJob:
    def __init__(self, config: Config, file_handler: FileHandler):
        self.config = config
        self.ssh_client = SSHClient(self.config.host, user=self.config.username)
        self.py_env = PythonEnvironment(config, ssh_client=self.ssh_client)
        self.file_handler = file_handler

    def _create_upload_slurm_script(self, main_script: str, params: ClusterJobBaseArguments, script_args: dict = None):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            sjob_filepath_local = os.path.join(tmp_dirname, self.config.jobscript)
            sjob_filepath_remote = os.path.join(self.config.workdir_path, self.config.jobscript)
            with open(sjob_filepath_local, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('\n')
                f.write(self.config.get_sbatch_header())
                f.write(self.py_env.get_header())
                f.write(self.config.get_env_header())
                f.write(self.py_env.get_run_command(main_script, script_args))
                f.write('\n')
            self.ssh_client.copy_file(sjob_filepath_local, sjob_filepath_remote)

            args_filepath_local = os.path.join(tmp_dirname, 'script_args.json')
            args_filepath_remote = self.config.args_file
            params.mode = 'local'
            params.save(args_filepath_local, with_reproducibility=False)
            self.ssh_client.copy_file(args_filepath_local, args_filepath_remote)

    def submit_job(self, main_script: str, params: ClusterJobBaseArguments):
        assert params.cluster_user is not None or 'You need to set --cluster-user'
        assert params.cluster_mail is not None or 'You need to set --cluster-mail'

        # checking whether we should do anything with data syncing
        if params.cluster_init:
            self.initialise()
        else:
            if params.upload_data:
                self.file_handler.upload_data()
            if params.upload_models:
                self.file_handler.cache_upload_models()

            self.file_handler.sync_code()

        print('Preparing and uploading slurm shell script...')
        self._create_upload_slurm_script(main_script=main_script, params=params)
        time.sleep(1)

        print('Triggering the job run...')
        out = self.ssh_client.run_command(f'sbatch {os.path.join(self.config.workdir_path, self.config.jobscript)}')
        job_id = list(out.stdout)
        if len(job_id) > 0:
            job_id = job_id[0]
            print(job_id)
            job_id = re.findall(r'(\d+)', job_id)[0]
            print('Follow outputs with:')
            print(f' $ tail -f {self.config.workdir_path}/{self.config.jobname}-{job_id}.out')
            print(f' $ tail -f {self.config.workdir_path}/{self.config.jobname}-{job_id}.err')
            print(f'Check job status with:')
            print(f' $ squeue -u {self.config.username}')
        else:
            print("Something went wrong (Couldn't read job id)...")
            print('\n'.join(job_id))

    def initialise(self):
        print('# Initialising cluster job (1/5) - Uploading requirements.txt...')
        self.file_handler.upload_requirements_txt()
        print('# Initialising cluster job (2/5) - Uploading code...')
        self.file_handler.sync_code(force_upload=True)
        print('# Initialising cluster job (3/5) - Preparing python env...')
        self.py_env.prepare_environment()
        print('# Initialising cluster job (4/5) - Uploading data...')
        self.file_handler.upload_data()
        print('# Initialising cluster job (5/5) - Uploading models...')
        self.file_handler.cache_upload_models()
        print('# Cluster job initialised!')
