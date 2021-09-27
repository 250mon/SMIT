import logging
import os


class Loggers:
    def __init__(self, name):
        self.name = name
        self.logger_dir = self._get_dir()

    def _get_dir(self):
        # path where the log file is created
        curr_dir = os.getcwd()
        logger_dir = os.path.join(curr_dir, 'custom_log', self.name)
        os.makedirs(logger_dir, exist_ok=True)
        return logger_dir

    # log to multiple files
    # a logger returned as create_logger is called
    # log files named logger_names are created under ./log/
    def create_logger(self,
                      logger_name,
                      f_log_lvl=logging.INFO,
                      c_log_lvl=logging.INFO):
        # create logger
        logger_name = self.name + '_' + logger_name
        logger = logging.getLogger(logger_name)
        logger.setLevel(f_log_lvl)
        # create file handler
        logger_filename = os.path.join(self.logger_dir, logger_name + '_log')
        fh = logging.FileHandler(logger_filename)
        fh.setLevel(f_log_lvl)
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(c_log_lvl)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s, %(name)-12s, %(levelname)s, %(message)s')
        fh.setFormatter(formatter)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    # log to a sigle file; but logger names may be different
    # in the application, call as follows;
    # logger1 = logging.getLogger('myapp.area1')
    # logger2 = logging.getLogger('myapp.area2')
    # a log file name log_file is created under ./log/
    def set_logging(self,
                    log_file,
                    f_log_lvl=logging.INFO,
                    c_log_lvl=logging.INFO):
        # set up logging to file - see previous section for more details
        filename = os.path.join(self.logger_dir,
                                self.name + '_' + log_file + '_log')
        logging.basicConfig(
            level=f_log_lvl,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=filename,
            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        ch = logging.StreamHandler()
        ch.setLevel(c_log_lvl)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        ch.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(ch)
