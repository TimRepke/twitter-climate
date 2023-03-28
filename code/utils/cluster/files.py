import re

from pssh.clients import SSHClient
from typing import List
import subprocess
import shlex
import os
from . import Config


class FileHandler:
    RE_FILE_LISTING = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) ((?:/[^/\n]+)+)")

    def __init__(self,
                 config: Config,
                 # local directory root all the following paths are based on
                 local_basepath: str,
                 # directories to look for code that should be uploaded relative to basepath
                 include_dirs: List[str],
                 # rules that we use to filter the list of files in prior parameter
                 # rules are applied relative to basepath
                 exclude_rules: List[str] = None,
                 # local path to the requirements.txt
                 requirements_txt: str = 'requirements.txt',
                 # list of (possibly large) data files to upload relative to basepath
                 data_files: List[str] = None,
                 # path to the local model cache
                 model_cache: str = None,
                 # models to cache
                 required_models: List[str] = None):
        self.config = config
        self.ssh_client = SSHClient(self.config.host, user=self.config.username)
        self.local_basepath = local_basepath
        self.requirements_txt = requirements_txt
        self.include_dirs = include_dirs
        self.exclude_rules = exclude_rules
        self.data_files = data_files

        if model_cache is None and required_models is not None:
            raise AttributeError('You need to specify model cache location.')
        self.model_cache = model_cache
        self.required_models = required_models

        if self.exclude_rules is None:
            self.exclude_rules = []
        self.exclude_rules.append(r'.*/?__pycache__.*/?')
        self.exclude_rules = [re.compile(r) for r in self.exclude_rules]

    def attach_ssh_client(self, ssh_client: SSHClient):
        self.ssh_client = ssh_client

    @classmethod
    def _parse_file_listing(cls, listing: List[str], basepath: str = ''):
        ret = []
        for line in listing:
            line_parts = cls.RE_FILE_LISTING.findall(line)

            if len(line_parts) == 1 and len(line_parts[0]) == 3:
                date = line_parts[0][0]
                time = line_parts[0][1]
                filepath = line_parts[0][2]
                ret.append((f'{date} {time}', filepath[len(basepath) + 1:]))
        return ret

    def _get_remote_file_listing(self, path: str, basepath: str = '') -> List[tuple]:
        directory_path = os.path.join(basepath, path)
        out = self.ssh_client.run_command(f'find {directory_path} -type f -print0 | '
                                          f'xargs -0 ls -l --time-style="+%F %T"')
        self.ssh_client.wait_finished(out)
        stderr = list(out.stderr)
        if len(stderr) > 0:
            # raise FileNotFoundError(f'Error retrieving listing for remote path: {directory_path}')
            return []
        stdout = list(out.stdout)
        return self._parse_file_listing(stdout, basepath)

    def _get_local_file_listing(self, path: str, basepath: str = '') -> List[tuple]:
        directory_path = os.path.join(basepath, path)
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f'This local file path does not exist: {directory_path}')
        p1 = subprocess.Popen(shlex.split(f'find {directory_path} -type f -print0'),
                              stdout=subprocess.PIPE)
        p2 = subprocess.Popen(shlex.split('xargs -0 ls -l --time-style="+%F %T"'),
                              stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        out = p2.communicate()[0]
        listing = self._parse_file_listing(out.decode().split('\n'), basepath)

        if self.exclude_rules is not None:
            def has_match(fp):
                return sum([bool(p.match(fp)) for p in self.exclude_rules]) > 0

            listing = [(dt, fp) for dt, fp in listing if not has_match(fp)]

        return listing

    def upload_requirements_txt(self):
        print('Uploading requirements.txt')
        local_req_txt = os.path.join(self.local_basepath, self.requirements_txt)
        remote_req_txt = os.path.join(self.config.workdir_path, 'requirements.txt')
        self.ssh_client.copy_file(local_req_txt, remote_req_txt)

    def sync_code(self, force_upload: bool = False):
        for directory in self.include_dirs:
            print('Fetching list of local files to upload...')
            local_listing = self._get_local_file_listing(directory, basepath=self.local_basepath)
            print('Fetching list of remote files to compare...')
            remote_listing = self._get_remote_file_listing(directory, basepath=self.config.codedir_path)
            remote_lookup = {fp: dt for dt, fp in remote_listing}
            for mod_date, local_filepath in local_listing:
                if force_upload or local_filepath not in remote_lookup or mod_date > remote_lookup[local_filepath]:
                    print(f'Uploading code file: {os.path.join(self.local_basepath, local_filepath)}')
                    self.ssh_client.copy_file(os.path.join(self.local_basepath, local_filepath),
                                              os.path.join(self.config.codedir_path, local_filepath))

    def upload_data(self):
        # TODO test if file needs uploading (eg by trying to download it and fetch SFTPIOError)
        if self.data_files is not None:
            for file in self.data_files:
                print(f'Uploading data file: {os.path.join(self.local_basepath, file)}')
                self.ssh_client.copy_file(os.path.join(self.local_basepath, file),
                                          os.path.join(self.config.datadir_path, file))

    def cache_upload_models(self):
        # TODO test if model needs uploading (eg by trying to download it and fetch SFTPIOError)
        if self.model_cache is not None:
            from utils.models import ModelCache
            model_cache = ModelCache(cache_dir=self.model_cache)
            for model_name in self.required_models:
                print(f'Uploading model "{model_name}"')
                model_cache.cache_model(model_name)
                self.ssh_client.copy_file(
                    str(model_cache.get_model_path(model_name)),
                    os.path.join(self.config.modeldir_path, model_cache.to_safe_name(model_name)),
                    recurse=True)
