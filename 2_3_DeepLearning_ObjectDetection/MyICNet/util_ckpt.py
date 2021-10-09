import tensorflow.compat.v1 as tf
import os


class CheckPtHandler:
    def __init__(self, ic_net):
        self.settings = ic_net.settings
        self.network = ic_net.network
        self.session = self.network.session
        self.saver = tf.train.Saver()
        self.ckpt_dir = self._get_dir()

    def _get_dir(self):
        os.makedirs(self.settings.ckpt_dir, exist_ok=True)
        return self.settings.ckpt_dir

    def save_ckpt(self, global_step):
        save_path = os.path.join(self.ckpt_dir, 'checkpoint')
        self.saver.save(self.session, save_path, global_step)
        print('\n The checkpoint has been created, epoch: {} \n'.format(global_step))

    # restore the most recent checkpoint
    def restore_ckpt(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_last_epoch = ckpt_name.rsplit('-', maxsplit=1)[1]
            start_epoch = int(ckpt_last_epoch) + 1
            print("---------------------------------------------------------")
            print(" Success to load checkpoint - {}".format(ckpt_name))
            print(" Session starts at epoch - {}".format(start_epoch))
            print("---------------------------------------------------------")
        else:
            start_epoch = 0
            print("**********************************************************")
            print("  [*] Failed to find a checkpoint - Start from the first")
            print(" Session starts at epoch - {}".format(start_epoch))
            print("**********************************************************")
        return start_epoch
