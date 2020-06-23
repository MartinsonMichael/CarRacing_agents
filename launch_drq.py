import hydra

from drq.train import Workspace


@hydra.main(config_path='drq/config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
