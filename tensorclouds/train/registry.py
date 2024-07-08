import yaml
import hydra_zen
import logging
import os
import shutil
from pathlib import Path

# import git

from .platform import Platform
import wandb


logger = logging.getLogger("aim.sdk.reporter")
logger.setLevel(logging.WARNING)


class Registry:
    """
    Abstract collection of platforms
    Registries can create new platforms for launching training
    """

    def __init__(self, project: str, base_path: str = None, source: str = None):
        if base_path is None:
            base_path = os.getenv("KHEIRON_REGISTRY_PATH")
        os.makedirs(base_path, exist_ok=True)

        if source is None:
            source = os.getcwd()

        path = os.path.join(base_path, project)
        os.makedirs(path, exist_ok=True)

        self.project = project
        self.path = Path(path)
        self.source = Path(source)

    def new_platform(
        self,
        cfg,
        tags=None,
        copy_repo=False,
    ) -> Platform:
        yml_cfg = hydra_zen.to_yaml(cfg)
        cfg_dict = yaml.safe_load(yml_cfg)
        run = wandb.init(
            project=self.project, dir=self.path, config=cfg_dict, tags=tags
        )

        name = run.name
        platform_dir = self.path / name
        os.makedirs(str(platform_dir), exist_ok=True)
        print(f"Copying Source Code to Platform -> {platform_dir}")

        def ignore_files(dir, files):
            print(f"Copying {dir}")
            ignore_list = [
                file
                for file in files
                if (
                    file.startswith(".")
                    or file in ("__pycache__",)
                    or dir[dir.rfind("/") + 1 :] in ("notebooks", "wandb")
                )
            ]
            return ignore_list

        if copy_repo:
            shutil.copytree(
                str(self.source / "model"),
                str(platform_dir / "code/"),
                ignore=ignore_files,
            )

        with open(f"{platform_dir}/config.yml", "w") as file:
            file.write(yml_cfg)
        print(f"New Model Image [{name}] at {platform_dir}")

        return Platform(name, platform_dir, run)

    def fetch_platform_names(self):
        dirs = os.listdir(str(self.path))
        dirs.sort(key=lambda x: -os.path.getmtime(os.path.join(self.path, x)))
        return dirs

    def get_platform(self, name: str, read_only=True) -> Platform:
        if name not in self.fetch_platform_names():
            raise ValueError(f"Platform {name} does not exist")
        if not os.path.exists(self.path / name / "config.yml"):
            raise ValueError(f"Platform {name} is missing config.yml")
        # if not os.path.exists(self.path / name / "code"):
        #     raise ValueError(f"Platform {name} is missing code directory"
        platform_path = self.path / name

        run = None  # Run(repo=self.repo, run_hash=hash, read_only=read_only)
        return Platform(hash, platform_path, run)

    def get_git_info(self):
        repo = git.Repo(self.source)
        return dict(
            branch=repo.active_branch.name,
            commit=repo.head.commit.hexsha,
            message=repo.head.commit.message,
            author=repo.head.commit.author.name,
            is_dirty=repo.is_dirty(),
        )
