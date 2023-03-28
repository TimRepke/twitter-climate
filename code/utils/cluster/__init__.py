from typing import List, Literal, Optional, Union
from dataclasses import dataclass, field
from tap import Tap
import os


class ClusterJobBaseArguments(Tap):
    mode: Literal['local', 'cluster'] = 'local'  # Whether to submit this as a cluster job or run locally.
    args_file: Optional[str] = None  # If this is set, cli args are scrapped and replaced with those from this file

    cluster_mail: Optional[str] = None  # email address for job notifications
    cluster_user: Optional[str] = None  # PIK username
    cluster_time: Optional[str] = '4:00:00'  # Time limit for the cluster job
    cluster_ram: Optional[str] = '20G'  # Memory limit for the cluster job
    cluster_gpu: bool = False  # Set this flag if the job needs GPU support
    cluster_n_cpus: int = 1  # The number of cores to be used
    cluster_jobname: str
    cluster_workdir: str

    upload_data: bool = False  # Set this flag to force data upload to cluster
    upload_models: bool = False  # Set this flag to force model upload to cluster
    cluster_init: bool = False  # Set this flag to initialise the cluster environment
    python_unbuffered: bool = False  # Set this flag to activate python unbuffered mode


@dataclass
class Config:
    # your username on the PIK Cluster
    username: str

    # will be used in naming the outputs
    jobname: str

    # working directory (remote)
    # will correspond to /p/tmp/{username}/{workdir}
    workdir: str

    # https://slurm.schedmd.com/sbatch.html#OPT_mail-user
    email_address: str

    # https://slurm.schedmd.com/sbatch.html#OPT_mem
    # Default units are megabytes. Different units can be specified using the suffix [K|M|G|T]
    memory: Union[int, str]

    # hostname of the cluster head
    host: str = 'cluster.pik-potsdam.de'

    # directory to place code (relative to {workdir})
    codedir: str = 'code'
    # directory to place data (relative to {workdir})
    datadir: str = 'data'
    # directory to place conda env (relative to {workdir})
    envdir: str = 'conda'
    # directory to place cached models (relative to {workdir})
    modeldir: str = 'models'

    # 2021.11: python 3.9
    # 2020.11: python 3.8.5
    # 2020.07: python 3.6, 3.7, 3.8
    anaconda: Literal['anaconda/2020.07', 'anaconda/2020.11', 'anaconda/2021.11'] = 'anaconda/2021.11'
    python: Literal['3.7', '3.8', '3.9'] = '3.9'

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#basic-concepts
    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#requesting-a-gpu-node
    partition: Literal['standard', 'gpu', 'largemem', 'io', 'ram'] = 'standard'
    gpu_type: Optional[Literal['v100', 'k40m']] = 'v100'  # NVidia Tesla K40m or Tesla V100
    n_gpus: Optional[Literal[1, 2]] = 1
    n_cpus: Optional[int] = 1

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#qos
    # short: max 24h; medium: max 7d; long: max 30d
    qos: Literal['short', 'medium', 'long'] = 'short'

    # https://gitlab.pik-potsdam.de/hpc-documentation/cluster-documentation/-/blob/master/Cluster%20User%20Guide.md#time-limits
    # Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds",
    # "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
    time_limit: Optional[str] = None

    # https://slurm.schedmd.com/sbatch.html#OPT_mail-type
    alerts: Optional[List[Literal['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL']]] = \
        field(default_factory=lambda: ['END', 'FAIL'])

    # environment variables
    env_vars: dict = None
    # environment variables that are only set before running the script (not for prepping)
    env_vars_run: dict = None

    std_out_file: str = None  # '%x-%j.out'
    std_err_file: str = None  # '%x-%j.err'
    python_unbuffered: bool = False

    @classmethod
    def from_args(cls, args: ClusterJobBaseArguments, **kwargs):
        config = cls(username=args.cluster_user,
                     email_address=args.cluster_mail,
                     jobname=args.cluster_jobname,
                     workdir=args.cluster_workdir,
                     memory=args.cluster_ram,
                     time_limit=args.cluster_time,
                     python_unbuffered=args.python_unbuffered,
                     n_cpus=args.cluster_n_cpus,
                     **kwargs)
        if args.cluster_gpu:
            config.partition = 'gpu'
        return config

    @property
    def std_out(self) -> str:
        if self.std_out_file is None:
            return f'{self.jobname}-%j.out'
        return self.std_out_file

    @property
    def std_err(self) -> str:
        if self.std_err_file is None:
            return f'{self.jobname}-%j.err'
        return self.std_err_file

    @property
    def workdir_path(self) -> str:
        return os.path.join('/p/tmp', self.username, self.workdir)

    @property
    def codedir_path(self) -> str:
        return os.path.join(self.workdir_path, self.codedir)

    @property
    def datadir_path(self) -> str:
        return os.path.join(self.workdir_path, self.datadir)

    @property
    def envdir_path(self) -> str:
        return os.path.join(self.workdir_path, self.envdir)

    @property
    def modeldir_path(self) -> str:
        return os.path.join(self.workdir_path, self.modeldir)

    @property
    def jobscript(self) -> str:
        return f'slurm_job_{self.jobname}.sh'

    @property
    def args_file(self) -> str:
        return os.path.join(self.workdir_path, f'args_{self.jobname}.json')

    def get_env_header(self, include_run_only: bool = True) -> str:
        ret = '\n'
        ret += '# Python environment\n'
        ret += f'export PYTHONPATH=$PYTHONPATH:{self.codedir_path}\n'
        if self.python_unbuffered:
            ret += f'export PYTHONUNBUFFERED=1\n'
        ret += '\n'
        ret += '# General environment variables\n'
        if self.env_vars is not None:
            for k, v in self.env_vars.items():
                ret += f'export {k}={v}\n'
        ret += '\n'
        ret += '# Environment variables for script\n'
        if include_run_only and self.env_vars_run is not None:
            for k, v in self.env_vars_run.items():
                ret += f'export {k}={v}\n'
        ret += '\n'
        return ret

    def get_sbatch_header(self) -> str:
        ret = '\n'
        ret += f'#SBATCH --mail-user={self.email_address}\n'
        ret += f'#SBATCH --workdir={self.workdir_path}\n'
        ret += f'#SBATCH --job-name={self.jobname}\n'
        ret += f'#SBATCH --qos={self.qos}\n'
        if self.time_limit is not None:
            ret += f'#SBATCH --time={self.time_limit}\n'
        ret += f'#SBATCH --partition={self.partition}\n'
        if self.partition == 'gpu':
            ret += f'#SBATCH --gres=gpu:{self.gpu_type}:{self.n_gpus}\n'
        ret += f'#SBATCH --nodes=1\n'
        ret += f'#SBATCH --cpus-per-task={self.n_cpus}\n'
        if self.n_cpus > 16:
            ret += f'#SBATCH --constraint=broadwell\n'
        ret += f'#SBATCH --mem={self.memory}\n'
        ret += f'#SBATCH --output={self.std_out}\n'
        ret += f'#SBATCH --error={self.std_err}\n'
        ret += f'#SBATCH --mail-type={",".join(self.alerts)}\n'
        ret += '\n'
        return ret
