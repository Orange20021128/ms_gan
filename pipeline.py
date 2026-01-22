# pipeline.py
from multiscale.ms_gan import MSGAN
from incomplete.recover_pipeline import recover_volume


class MSGANPipeline:
    def __init__(
        self,
        ms_gan: MSGAN,
        recover_model,
        device="cpu",
        n_class=4
    ):
        self.ms_gan = ms_gan
        self.recover_model = recover_model
        self.device = device
        self.n_class = n_class

    def run(self, borehole_volume):
        """
        borehole_volume: 稀疏 + 不完整
        """
        print("[Pipeline] Step 1: MS-GAN generation")
        vol = self.ms_gan.generate(borehole_volume)

        print("[Pipeline] Step 2: Recover incomplete voxels")
        vol = recover_volume(
            vol,
            self.recover_model,
            device=self.device,
            n_class=self.n_class
        )

        return vol
